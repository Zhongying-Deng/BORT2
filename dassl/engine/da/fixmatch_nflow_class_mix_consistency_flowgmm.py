import copy
import pdb
import itertools
import time
from random import uniform
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.distributions import (
        MultivariateNormal, Uniform, 
        TransformedDistribution, SigmoidTransform, Beta
        )
import torchvision.transforms as T
from thop import profile

from dassl.data import DataManager
from dassl.engine import TRAINER_REGISTRY, TrainerXU
from dassl.metrics import compute_accuracy
from dassl.data.transforms import build_transform
from dassl.engine.ssl import FixMatch
from dassl.evaluation import build_evaluator
from dassl.engine.trainer import SimpleNet
from dassl.utils import save_checkpoint, count_num_param, load_pretrained_weights, cutmix, mix_amplitude
from dassl.utils import (
        AffineConstantFlow, ActNorm, AffineHalfFlow, 
        SlowMAF, MAF, IAF, Invertible1x1Conv,
        NormalizingFlow, NormalizingFlowModel,
        )
from dassl.utils import NSF_AR, NSF_CL
from dassl.utils import RealNVP, RealNVPTabular, SSLGaussMixture, FlowLoss
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
            # For GeNRT, self.use_fmix is False
            self.use_fmix = False

    def forward(self, x, return_feature=False, domain_btcsize=None,
                use_fmix=False, label=None, noise=None):
        if self.use_fmix:
            f = self.backbone(x, domain_btcsize=domain_btcsize,
                    use_fmix=use_fmix, label=label)
        else:
            if self.head is None:
                f = self.backbone(x, noise=noise)
            else:
                f = self.backbone(x)
        if isinstance(f, tuple):
            f, fmix = f
        if self.head is not None:
            f = self.head(f)
            fmix[-1] = f
            if noise is not None:
                noise = torch.cat(noise, 0)
                lmda = Beta(0.1, 0.1).sample((f.size(0), 1))
                lmda = torch.max(lmda, 1. - lmda)
                lmda = lmda.to(f.device)
                feat = lmda * f + (1. - lmda) * noise
                feat = F.relu(feat)

        if self.classifier is None:
            return f, fmix

        y = self.classifier(f)

        if return_feature:
            return y, f, fmix

        return y, fmix


class FlowGMM(nn.Module):
    def __init__(self, dim, n_cls, means=None, inv_cov_stds=None, weights=None):
        super().__init__()
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.xpu.is_available():
            self.device = torch.device('xpu')
        else:
            self.device = torch.device('cpu')
        if means is not None:
            self.means = means
        else:
            self.means = torch.zeros((n_cls, dim)).to(self.device)
            for i in range(n_cls):
                self.means[i] = torch.randn(dim).to(self.device)
        self.net = RealNVPTabular(num_coupling_layers=6,in_dim=dim,hidden_dim=32,num_layers=1,dropout=True)
        self.prior = SSLGaussMixture(self.means, inv_cov_stds=inv_cov_stds, weights=weights, param_learnable=True)
        self.loss_fn = FlowLoss(self.prior)
        self.feat_cache = [[0.] for _ in range(n_cls)]
        self.initial_done = [False for _ in range(n_cls)]
        self.cache_len = [32 for _ in range(n_cls)]

    def forward(self, x, label=None, mask=None, return_unsup_loss=False):
        z1 = self.net(x)
        try:
            sldj = self.net.module.logdet()
        except:
            sldj = self.net.logdet()
        loss_unsup = self.loss_fn(z1, sldj=sldj)
        z_all = z1.reshape((len(z1), -1))
        if label is None:
            return loss_unsup, self.loss_fn.prior.class_logits(z_all)
        if mask is None:
            mask = torch.ones_like(label)
        labeled_mask = (mask != 0)
        if sum(labeled_mask) > 0:
            z_labeled = z_all[labeled_mask]
            y_labeled = label[labeled_mask]
            logits_all = self.loss_fn.prior.class_logits(z_all)
            logits_labeled = logits_all[labeled_mask]
            loss_nll = F.cross_entropy(logits_labeled, y_labeled)                
        else:
            loss_nll = torch.tensor(0.).to(self.device)
        if return_unsup_loss:
            return loss_nll, loss_unsup
        return loss_nll

    def predict(self, x):
        z1 = self.net(x)
        z_all = z1.reshape((len(z1), -1))
        pred = self.loss_fn.prior.class_logits(z_all)
        return pred
    
    def sample(self, batch_size, cls=None):
        with torch.no_grad():
            if cls is not None:
                z = self.prior.sample((batch_size,), gaussian_id=cls)
            else:
                z = self.prior.sample((batch_size,))
            try:
                x = self.net.module.inverse(z)
            except:
                x = self.net.inverse(z)

        return x

    def sample_classifier(self, x):
        z1 = self.net(x)
        z_all = z1.reshape((len(z1), -1))
        fc_sampled = self.loss_fn.prior.sample_classifier()
        pred = fc_sampled(z_all)
        return pred


@TRAINER_REGISTRY.register()
class FixMatchNFlowClassMixConsistencyFlowGMM(FixMatch):
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
        if cfg.TRAINER.NFLOW.EMA_MODEL_WEIGHTS:
            load_pretrained_weights(self.teacher, cfg.TRAINER.NFLOW.EMA_MODEL_WEIGHTS)
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
        self.weight_unsup_loss = cfg.TRAINER.NFLOW.WEIGHT_UNSUP_LOSS
        self.domain_aug = False
        self.feat_cache = [0. for _ in range(self.num_classes)]
        self.update_cache = [True for _ in range(self.num_classes)]
        self.feat_cache_used_len = [0 for _ in range(self.num_classes)]
        self.f_cache_len = 10
        print('cache length {}'.format(self.f_cache_len))
        self.adaptive_threshold = False
        
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
        print('dimension of features {}'.format(dim))
        self.model.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.model)))
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model('model', self.model, self.optim, self.sched)

        # Neural splines, coupling
        self.total_domain = len(self.cfg.DATASET.SOURCE_DOMAINS) + 1
        if (cfg.TRAINER.NFLOW.FLOW_MODEL_WEIGHTS or cfg.RESUME) and cfg.TRAINER.NFLOW.FLOW_MODEL_PRIORS:
            checkpoint = torch.load(cfg.TRAINER.NFLOW.FLOW_MODEL_PRIORS)
            mean = checkpoint['flow_mean']
            inv_cov_stds = checkpoint['flow_std']
            weights = checkpoint['flow_weights']
            self.nflow_model = FlowGMM(dim, self.num_classes, means=mean, inv_cov_stds=inv_cov_stds, weights=weights)
        else:
            print('Prior distribution of FlowGMM is randomly initialized!')
            self.nflow_model = FlowGMM(dim, self.num_classes)
        self.nflow_model.to(self.device)
        
        self.nflow_model.prior.means.requires_grad = True
        self.nflow_model.prior.weights.requires_grad = True
        self.nflow_model.prior.inv_cov_stds.requires_grad = True
        
        print('# params of NFlow model: {:,}'.format(count_num_param(self.nflow_model)))
        print('Use {} with lr {} for NFlow model'.format(cfg.TRAINER.NFLOW.OPTIM, cfg.TRAINER.NFLOW.LR))
        if cfg.TRAINER.NFLOW.FLOW_MODEL_WEIGHTS:
            load_pretrained_weights(self.nflow_model, cfg.TRAINER.NFLOW.FLOW_MODEL_WEIGHTS)
        if 'VisDA17' in cfg.DATASET.NAME or 'DomainNet' in cfg.DATASET.NAME:
            self.large_dataset = True
            self.optim_nflow = build_optimizer(self.nflow_model, cfg.OPTIM, 
                    lr=cfg.TRAINER.NFLOW.LR, optim=cfg.TRAINER.NFLOW.OPTIM, extra_params=None)
        else:
            self.large_dataset = False
            self.optim_nflow = build_optimizer(self.nflow_model, cfg.OPTIM, lr_mult=1., extra_params=None)
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
            if self.adaptive_threshold and self.epoch >= self.warmup:
                output_tmp = output_u[bs_src:]
                prob_u, _ = output_tmp.max(1)
                threshold = prob_u.mean() - prob_u.std()
                self.conf_thre = self.ema_alpha * self.conf_thre + (1. - self.ema_alpha) * threshold
            
            max_prob, label_u = output_u.max(1)
            mask_u = (max_prob >= self.conf_thre).float()
            self.num_samples += mask_u[bs_src:].sum()
            self.total_num += input_u.size(0)
            correct_ds_label = (label_u[bs_src:] == label_u_gt).float()
            self.label_ps_quality += correct_ds_label.sum()
            if not self.large_dataset:
                pred_nf = self.nflow_model.predict(intermediate_feat[-1][bs_src:, :].clone().detach())
                output_nflow = torch.softmax(pred_nf, 1)
            else:
                output_nflow = output_u[bs_src:, :]
            _, label_u_nflow = output_nflow.max(1)
            correct_ps_label = (label_u_nflow == label_u_gt).float()
            self.label_nflow_quality += correct_ps_label.sum()
            
            pred_ens = (output_nflow + output_u[bs_src:, :])/2.0
            _, label_u_ens = pred_ens.max(1)
            correct_ens_label = (label_u_ens == label_u_gt).float()
            self.label_ens_quality += correct_ens_label.sum()
            
        # forward normalizing flow model
        feat = intermediate_feat[-1].clone().detach()
        feat_input = feat[bs_src:, :]
        loss_ce, loss_unsup = self.nflow_model(feat_input, label=label_u[bs_src:], 
              mask=mask_u[bs_src:], return_unsup_loss=True)
        
        weight_unsup_nflow = get_consistency_weight(self.weight_unsup_loss, self.warmup, self.epoch)
        loss_flow = loss_ce + weight_unsup_nflow * loss_unsup
        
        if loss_flow != 0:
            self.model_backward_and_update(loss_flow, names='nflow_model')
        else:
            self.model_zero_grad(names='nflow_model')

        if (self.epoch + 1) < self.warmup:
            sampled_feat = None
        else:
            sampled_feat = []
            str_time = time.time()
            try:
                # not work
                sampled_noise = self.nflow_model.sample(input_u2.size(0), label_u)
                sampled_feat.append(sampled_noise)
            except:
                for i in range(input_u2.size(0)):
                     with torch.no_grad():
                          # class wise augmentation
                          d_idx = label_u[i]
                          if self.update_cache[d_idx]:
                              sampled_noise = self.nflow_model.sample(self.f_cache_len, d_idx)
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
               
        if sampled_feat is not None:
            sampled_feat_src = [torch.cat(sampled_feat, 0)[:bs_src]]
        else:
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
            
            _, label_u_mix, lam, rand_index = fmix
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
        if (self.epoch + 1) < self.warmup:
            loss_cons = torch.tensor(0.)
        else:
            feat_u = feat_all_u[-1].detach()
            str_time = time.time()
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
            'loss_flow': loss_flow.item(),
            'loss_flow_ce': loss_ce.item()
        }
        if self.adaptive_threshold:
            loss_summary['threshold'] = self.conf_thre
        if not ((self.epoch + 1) < self.warmup):
            loss_summary['weight_cons'] = weight_cons
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
    @torch.no_grad()
    def test(self):
        """A generic testing pipeline."""
        # display samples above the threshold
        print('samples above the threshold {}({}/{})'.format(
            float(self.num_samples ) / (self.total_num + 1e-7), self.num_samples, self.total_num))
        print('correct pseudo labels {}, correct nflow predictions {}, correct ensemble {}'.format(
            float(self.label_ps_quality) / (self.total_num + 1e-7), 
            float(self.label_nflow_quality) / (self.total_num + 1e-7),
            float(self.label_ens_quality) / (self.total_num + 1e-7))
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
                output_ens, _ = self.teacher(input)
            else:
                output, feat_all_u = self.model_inference(input)
                output_ens, _ = self.teacher(input)
            feat_u = feat_all_u[-1]
            pred_nf = self.nflow_model.predict(feat_u)
            output_teacher = torch.softmax(pred_nf, 1)
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
            tag_ema = 'nflow_{}/{}'.format(split, k)
            self.write_scalar(tag_ema, v, self.epoch)
        self.teacher.train()
        self.acc_teacher.append(results_teacher['accuracy'])
        
        for k, v in results_ens.items():
            tag_ens = 'ens_{}/{}'.format(split, k)
            self.write_scalar(tag_ens, v, self.epoch)
        self.acc_ens.append(results_ens['accuracy'])
        save_checkpoint(
                {
                    'flow_mean': self.nflow_model.prior.means,
                    'flow_std': self.nflow_model.prior.inv_cov_stds,
                    'flow_weights': self.nflow_model.prior.weights,
                    'epoch': self.epoch + 1
                },
                osp.join(self.output_dir, 'nflow_model_priors/'),
                is_best=False
                )

        if len(self.acc) > 1:
            if results['accuracy'] >= max(self.acc):
            # Save model
                self.save_model(self.epoch, self.output_dir, is_best=True)
            if results_ens['accuracy'] >= max(self.acc_ens):
               save_checkpoint(
                {
                    'state_dict': self.teacher.state_dict(),
                    'epoch': self.epoch + 1
                },
                osp.join(self.output_dir, 'ema/'),
                is_best=True
                )
        
        save_checkpoint(
            {
                'state_dict': self.teacher.state_dict(),
                'epoch': self.epoch + 1
                },
            osp.join(self.output_dir, 'ema/'),
            is_best=False
            )
 
        print('Until epoch {}, best accuracy of student model {}, nflow model {}, enssembled/EMA {}'.format(
            self.epoch+1, max(self.acc), max(self.acc_teacher), max(self.acc_ens)))

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
