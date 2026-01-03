import copy
import pdb
import itertools
from random import uniform
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
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
from dassl.utils import NSF_AR, NSF_CL
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
        self.feat_cache = [[0.] for _ in range(num_classes)]

    def forward(self, x, return_feature=False, domain_btcsize=None,
                use_fmix=False, label=None, noise=None, mask=None, weak_feat=None):
        
        if self.training and noise is None and use_fmix:
            noise = []
            for i in range(x.size(0)):
                x_i = weak_feat[i].unsqueeze(0)
                if mask[i] == 1:
                    d_idx = label[i]
                    if len(self.feat_cache[d_idx]) > 1:
                        noise.append(self.feat_cache[d_idx][0])
                        del self.feat_cache[d_idx][0]
                    else:
                        noise.append(x_i.clone().detach())
                else:
                    noise.append(x_i.clone().detach())

        if self.use_fmix:
            f = self.backbone(x, domain_btcsize=domain_btcsize,
                    use_fmix=use_fmix, label=label)
        else:
            f = self.backbone(x, noise=noise)
        if isinstance(f, tuple):
            f, fmix = f
            
        if self.training and noise is None and use_fmix:
            for i in range(f.size(0)):
                x_i = f[i].unsqueeze(0)
                if mask[i] == 1:
                    d_idx = label[i]
                    self.feat_cache[d_idx].append(x_i.clone().detach())
                    if not torch.is_tensor(self.feat_cache[d_idx][0]):
                        if self.feat_cache[d_idx][0] == 0.:
                            del self.feat_cache[d_idx][0]
                    if len(self.feat_cache[d_idx]) > 10:
                        del self.feat_cache[d_idx][0]

        if self.head is not None:
            f = self.head(f)

        if self.classifier is None:
            return f, fmix

        y = self.classifier(f)

        if return_feature:
            return y, f, fmix

        return y, fmix


class NFlowList(nn.Module):
    def __init__(self, dim, n_domains):
        super().__init__()
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.n_flows = n_domains
        self.feat_cache = [[0.] for _ in range(n_domains)]
        self.initial_done = [False for _ in range(n_domains)]
        self.cache_len = [32 for _ in range(n_domains)]
        nf_models = []
        for _ in range(n_domains):
            prior = TransformedDistribution(
                    Uniform(
                        torch.zeros(dim).to(self.device),
                        torch.ones(dim).to(self.device)
                        ),
                    SigmoidTransform().inv
                    ) # Logistic distribution
            nfs_flow = NSF_CL if True else NSF_AR
            flows = [nfs_flow(dim=dim, K=8, B=3, hidden_dim=16) for _ in range(3)]
            convs = [Invertible1x1Conv(dim=dim) for _ in flows]
            norms = [ActNorm(dim=dim) for _ in flows]
            flows = list(itertools.chain(*zip(norms, convs, flows)))
            nf_models.append(NormalizingFlowModel(prior, flows))
        
        # construct the normalizing flow model
        self.nflow_list = nn.ModuleList(nf_models)
        self.center = [0. for _ in range(n_domains)]

    def forward(self, x, domain_label=None, domain_bs=None, mask=None):
        # class-wise modeling
        loss = []
        for i in range(x.size(0)):
            # domain label is now class label
            if mask[i] == 1:
                d_idx = domain_label[i]
                x_i = x[i].unsqueeze(0)
                self.feat_cache[d_idx].append(x_i)
                if not torch.is_tensor(self.feat_cache[d_idx][0]):
                    if self.feat_cache[d_idx][0] == 0.:
                        del self.feat_cache[d_idx][0]
                if self.initial_done[d_idx]:
                    self.cache_len[d_idx] = 2
        class_label = domain_label.cpu().detach().numpy().tolist()
        label = set(class_label)
        for d_idx in label:
            if len(self.feat_cache[d_idx]) > self.cache_len[d_idx]:
                self.initial_done[d_idx] = True
                x_i = torch.cat(self.feat_cache[d_idx], dim=0)
                zs, prior_logprob, log_det = self.nflow_list[d_idx](x_i)
                del self.feat_cache[d_idx][1:]
                self.feat_cache[d_idx][0] = 0.
                logprob = prior_logprob + log_det
                loss_flow = -torch.mean(logprob) # NLL
                loss.append(loss_flow)
        if len(loss) == 0:
            loss.append(torch.tensor(0.))
        return loss

    def predict(self, x):
        prob = []
        for i in range(self.n_flows):
            zs, prior_logprob, log_det = self.nflow_list[i](x)
            logprob = prior_logprob + log_det
            if torch.is_tensor(self.center[i]):
                self.center[i] = self.center[i] * 0.99 + 0.01 * logprob.mean(dim=0).detach()
            else:
                # initialize center
                self.center[i] = logprob.mean(dim=0).detach()
            logprob = logprob - self.center[i]
            prob.append(logprob.unsqueeze(1))

        pred = torch.cat(prob, 1)
        pred = pred/100.
        return pred


@TRAINER_REGISTRY.register()
class FixMatchNFlowClassMixConsistencySrcMix(FixMatch):
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
        self.evaluator_ens = build_evaluator(cfg, lab2cname=self.dm.lab2cname)
        self.acc = []
        self.acc_teacher = []
        self.acc_ens = []
        self.num_samples = 0
        self.total_num = 0
        self.label_ps_quality = 0
        self.label_nflow_quality = 0
        self.label_ens_quality = 0
        self.lr_sched_type = cfg.OPTIM.LR_SCHEDULER
        self.x_btch_size = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
        self.u_btch_size = cfg.DATALOADER.TRAIN_U.BATCH_SIZE
        self.n_domain = len(self.cfg.DATASET.SOURCE_DOMAINS)
        batch_size = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
        self.split_batch = batch_size // self.n_domain
        self.weight_cons = cfg.TRAINER.CALOSS.WEIGHT_CON
        self.warmup = cfg.TRAINER.DDAIG.WARMUP
        self.domain_aug = False
        self.feat_cache = [0. for _ in range(self.num_classes)]
        self.update_cache = [True for _ in range(self.num_classes)]
        self.feat_cache_used_len = [0 for _ in range(self.num_classes)]
        self.f_cache_len = 10

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

        # Neural splines, coupling
        self.total_domain = len(self.cfg.DATASET.SOURCE_DOMAINS) + 1
        
        self.nflow_model = NFlowList(dim, self.num_classes)
        self.nflow_model.to(self.device)
        print('# params of NFlow model: {:,}'.format(count_num_param(self.nflow_model)))
        self.optim_nflow = build_optimizer(self.nflow_model, cfg.OPTIM, lr_mult=1.)
        self.sched_nflow = build_lr_scheduler(self.optim_nflow, cfg.OPTIM)
        self.register_model('nflow_model', self.nflow_model, self.optim_nflow, self.sched_nflow)

    def forward_backward(self, batch_x, batch_u):
        global_step = self.batch_idx + self.epoch * self.num_batches
        lmda = None
        parsed_data = self.parse_batch_train(batch_x, batch_u)
        input_x, input_x2, label_x, domain_xl, input_u, input_u2, label_u_gt = parsed_data

        domain_x = torch.split(domain_xl, self.split_batch, 0)
        domain_x = [d[0].item() for d in domain_x]
        domain_x.append(max(domain_x)+1)
        
        input_u1 = torch.cat([input_x, input_u], 0)
        input_u2 = torch.cat([input_x2, input_u2], 0)
        domain_btch_size = [self.split_batch for _ in range(self.n_domain)]

        use_fmix = False
        bs_src = input_x.size(0)
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
            self.num_samples += mask_u[bs_src:].sum()
            self.total_num += input_u.size(0)
            correct_ds_label = (label_u[bs_src:] == label_u_gt).float()
            self.label_ps_quality += correct_ds_label.sum()
            
            pred_nf = self.nflow_model.predict(intermediate_feat[-1][bs_src:, :])
            output_nflow = torch.softmax(pred_nf, 1)
            _, label_u_nflow = output_nflow.max(1)
            correct_ps_label = (label_u_nflow == label_u_gt).float()
            self.label_nflow_quality += correct_ps_label.sum()
            
            pred_ens = (output_nflow + output_u[bs_src:, :])/2.0
            _, label_u_ens = pred_ens.max(1)
            correct_ens_label = (label_u_ens == label_u_gt).float()
            self.label_ens_quality += correct_ens_label.sum()
            
        # forward normalizing flow model
        feat = intermediate_feat[-1].clone().detach()#[0]
        feat_input = feat[bs_src:, :] #Variable(feat, requires_grad=True)
        loss_all = self.nflow_model(feat_input, domain_label=label_u[bs_src:], 
              domain_bs=domain_btch_size, mask=mask_u[bs_src:])
        
        loss_flow = sum(loss_all) / len(loss_all)
        if loss_flow != 0:
            self.model_backward_and_update(loss_flow, names='nflow_model')
        else:
            self.model_zero_grad(names='nflow_model')

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
                              sampled_noise = self.nflow_model.nflow_list[d_idx].sample(self.f_cache_len)
                              self.feat_cache[d_idx] = sampled_noise[-1].detach()
                          f_idx = self.feat_cache_used_len[d_idx]
                          feat1 = self.feat_cache[d_idx][f_idx, :]
                          sampled_feat.append(feat1.unsqueeze(0))
                          self.feat_cache_used_len[d_idx] = self.feat_cache_used_len[d_idx] + 1
                          if self.feat_cache_used_len[d_idx] > (self.f_cache_len - 1):
                              self.feat_cache_used_len[d_idx] = 0
                              self.update_cache[d_idx] = True
                          else:
                              self.update_cache[d_idx] = False
        
        if sampled_feat is not None:
            sampled_feat_src = None
        else:
            sampled_feat_src = None
        output_x, _ = self.model(input_x, use_fmix=True, label=label_u[:bs_src],
                                  mask=mask_u[:bs_src], weak_feat=intermediate_feat[-1][:bs_src])#noise=sampled_feat_src)
        
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
            pred_nf = self.nflow_model.predict(feat_u[bs_src:])
            pred_nf = torch.softmax(pred_nf, 1)

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
            'loss_con': loss_cons.item(),
            'loss_flow': loss_flow.item()
        }

        if self.lr_sched_type != 'fixmatch':
            if (self.batch_idx + 1) == self.num_batches:
                self.update_lr()
        else:
            self.update_lr()

        return loss_summary

    def nflow_pred(self, x):
        prob = []
        for i in range(self.num_classes):
            zs, prior_logprob, log_det = self.nflow_model[i](x)
            logprob = prior_logprob + log_det
            prob.append(logprob.unsqueeze(1))
        
        pred = torch.cat(prob, 1)
        pdb.set_trace()
        pred = torch.softmax(pred, 1)
        return pred

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

    @torch.no_grad()
    def test(self):
        """A generic testing pipeline."""
        # display samples above the threshold
        print('samples above the threshold {}({}/{})'.format(
            float(self.num_samples ) / self.total_num, self.num_samples, self.total_num))
        print('correct pseudo labels {}, correct nflow predictions {}, correct ensemble {}'.format(
            float(self.label_ps_quality) / self.total_num, 
            float(self.label_nflow_quality) / self.total_num,
            float(self.label_ens_quality) / self.total_num)
            )
        self.num_samples = 0
        self.total_num = 0
        self.label_nflow_quality = 0
        self.label_ps_quality = 0
        self.label_ens_quality = 0

        self.set_model_mode('eval')
        self.teacher.eval()
        self.evaluator.reset()
        self.evaluator_teacher.reset()
        self.evaluator_ens.reset()

        split = self.cfg.TEST.SPLIT
        print('Do evaluation on {} set'.format(split))
        data_loader = self.val_loader if split == 'val' else self.test_loader
        assert data_loader is not None
        weight_cons = get_consistency_weight(self.weight_cons, self.warmup, self.epoch)

        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            if self.use_fmix:
                output, _ = self.model_inference(input)
            else:
                output, feat_all_u = self.model_inference(input)
            feat_u = feat_all_u[-1]
            pred_nf = self.nflow_model.predict(feat_u)
            output_teacher = torch.softmax(pred_nf, 1)
            output_ens = (output + weight_cons * pred_nf) / (1 + weight_cons)
            self.evaluator.process(output, label)
            self.evaluator_teacher.process(output_teacher, label)
            self.evaluator_ens.process(output_ens, label)

        results = self.evaluator.evaluate()
        results_teacher = self.evaluator_teacher.evaluate()
        results_ens = self.evaluator_ens.evaluate()

        for k, v in results.items():
            tag = '{}/{}'.format(split, k)
            self.write_scalar(tag, v, self.epoch)
        self.acc.append(results['accuracy'])

        for k, v in results_teacher.items():
            tag_ema = 'ema_{}/{}'.format(split, k)
            self.write_scalar(tag_ema, v, self.epoch)
        self.teacher.train()
        self.acc_teacher.append(results_teacher['accuracy'])
        
        for k, v in results_ens.items():
            tag_ens = 'ens_{}/{}'.format(split, k)
            self.write_scalar(tag_ens, v, self.epoch)
        self.acc_ens.append(results_ens['accuracy'])

        print('Until epoch {}, best accuracy of student model {}, nflow model {}, enssembled {}'.format(
            self.epoch, max(self.acc), max(self.acc_teacher), max(self.acc_ens)))

    def parse_batch_train(self, batch_x, batch_u):
        input_x = batch_x['img']
        input_x2 = batch_x['img2']
        label_x = batch_x['label']
        domain_x = batch_x['domain']
        input_u = batch_u['img']
        input_u2 = batch_u['img2']
        label_u = batch_u['label']

        input_x = input_x.to(self.device)
        input_x2 = input_x2.to(self.device)
        label_x = label_x.to(self.device)
        input_u = input_u.to(self.device)
        input_u2 = input_u2.to(self.device)

        label_u = label_u.to(self.device)
        return input_x, input_x2, label_x, domain_x, input_u, input_u2, label_u
