import copy
import itertools
from random import uniform
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import MultivariateNormal, Uniform, TransformedDistribution, SigmoidTransform
import torchvision.transforms as T

from dassl.data import DataManager
from dassl.engine import TRAINER_REGISTRY, TrainerXU
from dassl.metrics import compute_accuracy
from dassl.data.transforms import build_transform
from dassl.engine.ssl import FixMatch
from dassl.evaluation import build_evaluator
from dassl.engine.trainer import SimpleNet
from dassl.utils import count_num_param, cutmix, mix_amplitude
from dassl.utils import (
        AffineConstantFlow, ActNorm, AffineHalfFlow, 
        SlowMAF, MAF, IAF, Invertible1x1Conv,
        NormalizingFlow, NormalizingFlowModel,
        )
from dassl.utils import NSF_AR, NSF_CL, load_pretrained_weights
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.modeling import build_head, build_backbone

def domain_discrepancy(out1, out2, loss_type):
    def huber_loss(e, d=1):
        t =torch.abs(e)
        ret = torch.where(t < d, 0.5 * t ** 2, d * (t - 0.5 * d))
        return torch.mean(ret)

    diff = out1 - out2
    if loss_type == 'L1':
        loss = torch.mean(torch.abs(diff))
    elif loss_type == 'Huber':
        loss = huber_loss(diff)
    else:
        loss = torch.mean(diff*diff)
    return loss


def get_consistency_weight(consistency_scale, consistency_rampup, epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    def sigmoid_rampup(current, rampup_length):
        """Exponential rampup from https://arxiv.org/abs/1610.02242"""
        import numpy as np
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))

    consistency_weight = consistency_scale * sigmoid_rampup(epoch, consistency_rampup)
    return consistency_weight


class DomainBasedNet(SimpleNet):
    """A simple neural network composed of a CNN backbone
    and optionally a head such as mlp for classification.
    """
    def __init__(self, cfg, model_cfg, num_classes, **kwargs):
        super().__init__(cfg, model_cfg, num_classes, **kwargs)
        self.backbone = build_backbone(
                model_cfg.BACKBONE.NAME,
                verbose=cfg.VERBOSE,
                pretrained=model_cfg.BACKBONE.PRETRAINED,
                **kwargs
                )
        if 'fmix' in model_cfg.BACKBONE.NAME:
            self.use_fmix = True
        else:
            self.use_fmix = False

    def forward(self, x, return_feature=False, domain_btcsize=None,
                use_fmix=False, label=None, noise=None):
        if self.use_fmix:
            f = self.backbone(x, domain_btcsize=domain_btcsize,
                    use_fmix=use_fmix, label=label)
        else:
            f = self.backbone(x, noise=noise)
        if isinstance(f, tuple):
            f, fmix = f
        if self.head is not None:
            f = self.head(f)

        if self.classifier is None:
            return f, fmix

        y = self.classifier(f)

        if return_feature:
            return y, f, fmix

        return y, fmix


@TRAINER_REGISTRY.register()
class FixMatchGaussianClassMixConsistency(FixMatch):
    """FixMatch: Simplifying Semi-Supervised Learning with
    Consistency and Confidence.

    https://arxiv.org/abs/2001.07685.

    FixMatch with Domain labels, sharing config file with base FixMatch,
    also with FeatMix and MixStyle

    Sangdoo Yun et al. 'CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features'. ICCV 2019
    Kaiyang Zhou, Yongxin Yang, Yu Qiao, and Tao Xiang. Domain Generalization with MixStyle. ICLR, 2021
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.weight_u = cfg.TRAINER.FIXMATCH.WEIGHT_U
        self.conf_thre = cfg.TRAINER.FIXMATCH.CONF_THRE
        self.ema_alpha = cfg.TRAINER.FIXMATCH.EMA_ALPHA

        self.teacher = copy.deepcopy(self.model)
        self.teacher.train()
        for param in self.teacher.parameters():
            param.requires_grad_(False)
        self.evaluator_teacher = build_evaluator(cfg, lab2cname=self.dm.lab2cname)
        self.acc = []
        self.acc_teacher = []
        self.num_samples = 0
        self.total_num = 0
        self.lr_sched_type = cfg.OPTIM.LR_SCHEDULER
        self.x_btch_size = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
        self.u_btch_size = cfg.DATALOADER.TRAIN_U.BATCH_SIZE
        self.n_domain = len(self.cfg.DATASET.SOURCE_DOMAINS)
        batch_size = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
        self.split_batch = batch_size // self.n_domain
        self.weight_cons = cfg.TRAINER.CALOSS.WEIGHT_CON
        self.warmup = cfg.TRAINER.DDAIG.WARMUP
        assert self.warmup > 0
        self.domain_aug = False
        self.feat_cache = [0. for _ in range(self.num_classes)]
        self.update_cache = [True for _ in range(self.num_classes)]
        self.feat_cache_used_len = [0 for _ in range(self.num_classes)]
        self.f_cache_len = 10
        self.run_mean = [0. for _ in range(self.num_classes)]
        self.run_var = [0. for _ in range(self.num_classes)]

    def build_model(self):
        cfg = self.cfg
        print('Building model')
        self.fmix_cfg = cfg.TRAINER.FEATMIX.CONFIG
        self.fmix_prob = cfg.TRAINER.FEATMIX.PROB
        self.fmix_beta = cfg.TRAINER.FEATMIX.BETA
        assert sum(self.fmix_cfg)==1, 'only one element in TRAINER.FEATMIX.CONFIG can be 1'
        self.model = DomainBasedNet(
                cfg, cfg.MODEL, 
                self.num_classes, 
                fmix_cfg=self.fmix_cfg, 
                fmix_beta=self.fmix_beta, 
                fmix_prob=self.fmix_prob
                )
        if 'fmix' in cfg.MODEL.BACKBONE.NAME:
            self.use_fmix = True
        else:
            self.use_fmix = False

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
        dim = self.model.fdim
        self.model.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.model)))
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model('model', self.model, self.optim, self.sched)

    def forward_backward(self, batch_x, batch_u):
        global_step = self.batch_idx + self.epoch * self.num_batches
        lmda = None
        parsed_data = self.parse_batch_train(batch_x, batch_u)
        input_x, input_x2, label_x, domain_xl, input_u, input_u2 = parsed_data

        domain_x = torch.split(domain_xl, self.split_batch, 0)
        domain_x = [d[0].item() for d in domain_x]
        domain_x.append(max(domain_x)+1)
        input_u1 = torch.cat([input_x, input_u], 0)
        input_u2 = torch.cat([input_x2, input_u2], 0)
        domain_btch_size = [self.split_batch for _ in range(self.n_domain)]

        use_fmix = False
        # Generate artificial label
        with torch.no_grad():
            # when generating pseudo labels, we do not use MixStyle
            domain_btch_size.append(input_u.size(0))
            if self.use_fmix:
                logit, _ = self.model(input_u1, use_fmix=use_fmix,
                        domain_btcsize=domain_btch_size)
            else:
                logit, intermediate_feat = self.model(input_u1)
            output_u = F.softmax(logit, 1)

            max_prob, label_u = output_u.max(1)
            mask_u = (max_prob >= self.conf_thre).float()
            self.num_samples += mask_u.sum()
            self.total_num += label_u.size()[0]
            
        start_idx = input_x.size(0)
        feat = intermediate_feat[-1].clone().detach()
        self.update_running_mean_var(
                feat[start_idx:, :], 
                label_u[start_idx:], 
                mask_u[start_idx:]
                )
        if self.epoch < self.warmup:
            sampled_feat = None
        else:
            sampled_feat = []
            for i in range(input_u2.size(0)):
                if self.domain_aug:
                    d_idx = domain_x[i]
                    # sampled_noise is a list with length=10
                    sampled_noise = self.nflow_model.nflow_list[d_idx].sample(domain_btch_size[i])
                    sampled_feat.append(sampled_noise[-1].detach())
                    sampled_feat = sampled_feat[::-1]
                else:
                    with torch.no_grad():
                        # class wise augmentation
                        d_idx = label_u[i]
                        if self.update_cache[d_idx]:
                            normal_dist = MultivariateNormal(
                                    self.run_mean[d_idx], 
                                    torch.diag(self.run_var[d_idx])
                                    )
                            sampled_noise = normal_dist.sample((self.f_cache_len,))
                            self.feat_cache[d_idx] = sampled_noise.detach()
                        f_idx = self.feat_cache_used_len[d_idx]
                        feat1 = self.feat_cache[d_idx][f_idx, :]
                        sampled_feat.append(feat1.unsqueeze(0))
                        self.feat_cache_used_len[d_idx] = self.feat_cache_used_len[d_idx] + 1
                        if self.feat_cache_used_len[d_idx] > (self.f_cache_len - 1):
                            self.feat_cache_used_len[d_idx] = 0
                            self.update_cache[d_idx] = True
                        else:
                            self.update_cache[d_idx] = False

        bs_src = input_x.size(0)
        sampled_feat_src = None
        if self.use_fmix:
            output_x, _ = self.model(input_x, use_fmix=False, noise=sampled_feat_src)  # use_mixstyle)
        else:
            output_x, _ = self.model(input_x, noise=sampled_feat_src)
        loss_x = F.cross_entropy(output_x, label_x)

        # Unsupervised loss
        use_fmix = True
        if self.use_fmix:
            output_u, fmix = self.model(input_u2, use_fmix=use_fmix,
                                  domain_btcsize=domain_btch_size, label=label_u, noise=sampled_feat)
            
            _, label_u_mix, lam, rand_index = fmix #[idx]
            loss_u1 = F.cross_entropy(output_u, label_u, reduction='none')
            loss_u1 = lam * (loss_u1 * mask_u).mean()
            loss_u_mix = F.cross_entropy(output_u, label_u_mix, reduction='none')
            loss_u_mix = (1. - lam) * (loss_u_mix * mask_u[rand_index]).mean()
            loss_u = loss_u_mix + loss_u1
        else:
            if sampled_feat is not None:
                output_u, feat_all_u = self.model(input_u2, noise=sampled_feat)
            else:
                output_u, feat_all_u = self.model(input_u2, noise=None)
            loss_u = F.cross_entropy(output_u, label_u, reduction='none')
            loss_u = (loss_u * mask_u).mean()
        if self.epoch < 2 * self.warmup:
            loss_cons = torch.tensor(0.)
        else:
            feat_u = feat_all_u[-1].detach()
            pred_nf = self.gaussian_pred(feat_u[bs_src:])

            pred_c = F.softmax(output_u[bs_src:], 1)
           
            weight_cons = get_consistency_weight(self.weight_cons, self.warmup, self.epoch - 2 * self.warmup)
            loss_cons = weight_cons * domain_discrepancy(pred_nf, pred_c, 'L2')

        loss = loss_x + loss_u * self.weight_u + loss_cons
        self.model_backward_and_update(loss, names='model')

        ema_alpha = min(1 - 1 / (global_step+1), self.ema_alpha)
        self.ema_model_update(self.model, self.teacher, ema_alpha)

        loss_summary = {
            'loss_x': loss_x.item(),
            'acc_x': compute_accuracy(output_x, label_x)[0].item(),
            'loss_u': loss_u.item(),
            'acc_u': compute_accuracy(output_u, label_u)[0].item(),
            'loss_con': loss_cons.item()
        }

        if self.lr_sched_type != 'fixmatch':
            if (self.batch_idx + 1) == self.num_batches:
                self.update_lr()
        else:
            self.update_lr()

        return loss_summary

    def ema_model_update(self, model, ema, decay):
        ema_has_module = hasattr(ema, 'module')
        needs_module = hasattr(model, 'module') and not ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            for k, ema_v in ema.state_dict().items():
                if needs_module:
                    k = 'module.' + k
                model_v = msd[k].detach()
                ema_v.copy_(ema_v * decay + (1. - decay) * model_v)

    def update_running_mean_var(self, x, label, mask, ema=0.9):
        for i in range(x.size(0)):
            if mask[i] == 1:
                label_i = label[i]
                self.run_mean[label_i] = self.run_mean[label_i] * ema + (1-ema) * x[i, :]
                diff = x[i, :] - self.run_mean[label_i]
                std = torch.sqrt(diff*diff) + 1e-6
                self.run_var[label_i] = self.run_var[label_i] * ema + (1-ema) * std

    def gaussian_pred(self, x):
        prob = []
        for i in range(self.num_classes):
            normal_dist = MultivariateNormal(
                    self.run_mean[i], 
                    torch.diag(self.run_var[i])
                    )
            logprob = normal_dist.log_prob(x) / x.size(1)
            prob.append(logprob.unsqueeze(1))
        
        pred = torch.cat(prob, 1)
        pred = torch.softmax(pred, 1)
        return pred

    @torch.no_grad()
    def test(self):
        """A generic testing pipeline."""
        # display samples above the threshold
        print('samples above the threshold {}({}/{})'.format(
            float(self.num_samples ) / self.total_num, self.num_samples, self.total_num))
        self.num_samples = 0
        self.total_num = 0

        self.set_model_mode('eval')
        self.teacher.eval()
        self.evaluator.reset()
        self.evaluator_teacher.reset()

        split = self.cfg.TEST.SPLIT
        print('Do evaluation on {} set'.format(split))
        data_loader = self.val_loader if split == 'val' else self.test_loader
        assert data_loader is not None

        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            if self.use_fmix:
                output, _ = self.model_inference(input)
                output_teacher, _ = self.teacher(input)
            else:
                output, _ = self.model_inference(input)
                output_teacher, _ = self.teacher(input)
            self.evaluator.process(output, label)
            self.evaluator_teacher.process(output_teacher, label)

        results = self.evaluator.evaluate()
        results_teacher = self.evaluator_teacher.evaluate()

        for k, v in results.items():
            tag = '{}/{}'.format(split, k)
            self.write_scalar(tag, v, self.epoch)
        self.acc.append(results['accuracy'])

        for k, v in results_teacher.items():
            tag_ema = 'ema_{}/{}'.format(split, k)
            self.write_scalar(tag_ema, v, self.epoch)
        self.teacher.train()
        self.acc_teacher.append(results_teacher['accuracy'])
        print('Until epoch {}, best accuracy of student model {}, teacher model {}'.format(self.epoch, max(self.acc), max(self.acc_teacher)))

    def parse_batch_train(self, batch_x, batch_u):
        input_x = batch_x['img']
        input_x2 = batch_x['img2']
        label_x = batch_x['label']
        domain_x = batch_x['domain']
        input_u = batch_u['img']
        input_u2 = batch_u['img2']

        input_x = input_x.to(self.device)
        input_x2 = input_x2.to(self.device)
        label_x = label_x.to(self.device)
        input_u = input_u.to(self.device)
        input_u2 = input_u2.to(self.device)

        return input_x, input_x2, label_x, domain_x, input_u, input_u2

