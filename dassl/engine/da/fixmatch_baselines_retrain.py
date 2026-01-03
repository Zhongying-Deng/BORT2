import copy
from random import uniform
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as T

from dassl.data import DataManager
from dassl.engine import TRAINER_REGISTRY, TrainerXU
from dassl.metrics import compute_accuracy
from dassl.data.transforms import build_transform
from dassl.engine.ssl import FixMatch
from dassl.evaluation import build_evaluator
from dassl.engine.trainer import SimpleNet
from dassl.utils import count_num_param, load_pretrained_weights, cutmix, mix_amplitude
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.modeling import build_head, build_backbone


def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


class PairClassifiers(nn.Module):

    def __init__(self, fdim, num_classes):
        super().__init__()
        self.c1 = nn.Linear(fdim, num_classes)
        self.c2 = nn.Linear(fdim, num_classes)

    def forward(self, x):
        z1 = self.c1(x)
        if not self.training:
            return z1
        z2 = self.c2(x)
        return z1, z2


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
        if 'ms' in model_cfg.BACKBONE.NAME:
            self.use_mixstyle = True
        else:
            self.use_mixstyle = False

    def forward(self, x, return_feature=False, domain_btcsize=None,
                use_mixstyle=True, lmda=None):
        if self.use_mixstyle:
            f = self.backbone(x, domain_btcsize=domain_btcsize,
                    use_mixstyle=use_mixstyle, lmda=lmda)
        else:
            f = self.backbone(x)
        if isinstance(f, tuple):
            f, ms = f
        if self.head is not None:
            f = self.head(f)

        if self.classifier is None:
            return f

        y = self.classifier(f)

        if return_feature:
            return y, f

        return y


@TRAINER_REGISTRY.register()
class FixMatchBaselinesRetrain(FixMatch):
    """FixMatch: Simplifying Semi-Supervised Learning with
    Consistency and Confidence.

    https://arxiv.org/abs/2001.07685.

    FixMatch with Domain labels, sharing config file with base FixMatch,
    also with CutMix and MixStyle

    Sangdoo Yun et al. 'CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features'. ICCV 2019
    Kaiyang Zhou, Yongxin Yang, Yu Qiao, and Tao Xiang. Domain Generalization with MixStyle. ICLR, 2021
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.weight_u = cfg.TRAINER.FIXMATCH.WEIGHT_U
        self.conf_thre = cfg.TRAINER.FIXMATCH.CONF_THRE
        self.ema_alpha = cfg.TRAINER.FIXMATCH.EMA_ALPHA
        self.thre_param = - torch.log(1/torch.tensor(self.conf_thre) - 1)

        self.evaluator_new_model = build_evaluator(cfg, lab2cname=self.dm.lab2cname)
        self.acc = []
        self.acc_teacher = []
        self.acc_new_model = []
        self.num_samples = 0
        self.label_ps_quality = 0
        self.total_num = 0
        self.lr_sched_type = cfg.OPTIM.LR_SCHEDULER
        self.x_btch_size = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
        self.u_btch_size = cfg.DATALOADER.TRAIN_U.BATCH_SIZE
        self.n_domain = len(self.cfg.DATASET.SOURCE_DOMAINS)
        batch_size = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
        self.split_batch = batch_size // self.n_domain
        self.cutmix_prob = cfg.TRAINER.CUTMIX.PROB
        self.cutmix_beta = cfg.TRAINER.CUTMIX.BETA
        self.built_new_optim = False
        self.high_ratio_tar = False
        self.retrain_ratio = cfg.TRAINER.RETRAIN.RATIO
        self.retrain_epoch = cfg.TRAINER.RETRAIN.EPOCH
        self.finetune = True
        self.thre_param = cfg.TRAINER.FIXMATCH.CONF_THRE
        self.test(False)

    def build_model(self):
        self.method = self.cfg.TRAINER.RETRAIN.METHOD
        cfg = self.cfg
        print('Building model')
        ms_cfg = cfg.TRAINER.MIXSTYLE.CONFIG
        self.model = DomainBasedNet(cfg, cfg.MODEL, 0, ms_cfg=ms_cfg, style_g_train=True)
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
        else:
            assert False, 'MODEL.INIT_WEIGHTS is empty. Initial weights not provided.'
        fdim = self.model.fdim
        self.model.to(self.device)
        if self.method == 'MCD':
            print('Building C')
            self.C = nn.Linear(fdim, self.num_classes)
            if cfg.MODEL.INIT_WEIGHTS:
                weights_c = cfg.MODEL.INIT_WEIGHTS
                weights_c = weights_c.replace('F', 'C1')
        elif self.method == 'M3SDA':
            self.C = nn.ModuleList(
                    [
                        PairClassifiers(fdim, self.num_classes)
                        for _ in range(self.dm.num_source_domains)
                        ]
                    )
            if cfg.MODEL.INIT_WEIGHTS:
                weights_c = cfg.MODEL.INIT_WEIGHTS
                weights_c = weights_c.replace('F', 'C')
        else:
            weights_c = cfg.MODEL.INIT_WEIGHTS
        load_pretrained_weights(self.C, weights_c)
        self.C.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.model)))
        self.optim = build_optimizer(self.model, cfg.OPTIM, lr_mult=0.1)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model('model', self.model, self.optim, self.sched)

        print('# params: {:,}'.format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.OPTIM, lr_mult=0.1)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.OPTIM)
        self.register_model('C', self.C, self.optim_C, self.sched_C)
        # load pseudo labeling function
        self.F_labeler = SimpleNet(cfg, cfg.MODEL, 0)
        if self.method == 'MCD':
            self.C_labeler = nn.Linear(fdim, self.num_classes)
        elif self.method == 'M3SDA':
            self.C_labeler = nn.ModuleList(
                [
                    PairClassifiers(fdim, self.num_classes)
                    for _ in range(self.dm.num_source_domains)
                ]
            )
        else:
            pass
        self.F_labeler.to(self.device)
        self.C_labeler.to(self.device)
        model_weights = copy.deepcopy(self.model.state_dict())
        self.F_labeler.load_state_dict(model_weights)
        model_weights = copy.deepcopy(self.C.state_dict())
        self.C_labeler.load_state_dict(model_weights)
        self.F_labeler.eval()
        self.F_labeler.apply(fix_bn)
        for param in self.F_labeler.parameters():
            param.requires_grad_(False)
        self.C_labeler.eval()
        self.C_labeler.apply(fix_bn)
        for param in self.C_labeler.parameters():
            param.requires_grad_(False)
        self.built_new_optim = True

    def forward_backward(self, batch_x, batch_u):
        global_step = self.batch_idx + self.epoch * self.num_batches
        ema_alpha = min(1 - 1 / (global_step+1), self.ema_alpha)
        lmda = None
        parsed_data = self.parse_batch_train(batch_x, batch_u)
        input_x, input_x2, label_x, input_u, input_u2, label_u_gt = parsed_data
        input_u1 = input_u
        start_idx = 0
        domain_btch_size = [self.split_batch for _ in range(self.n_domain)]
           
        use_mixstyle = False
        # Generate artificial label
        with torch.no_grad():
            # when generating pseudo labels, we do not use MixStyle
            domain_btch_size.append(input_u.size(0))
            output_u = F.softmax(self.labeler_inference(input_u1), 1)
            del domain_btch_size[-1]
            output_tmp = output_u[start_idx:]
            prob_u, _ = output_tmp.max(1)
            threshold = prob_u.mean() - prob_u.std()
            if global_step == 0:
                self.thre_param = prob_u.mean() + prob_u.std()
            self.thre_param = self.ema_alpha * self.thre_param + (1. - self.ema_alpha) * threshold
            self.conf_thre = self.thre_param

            max_prob, label_u = output_u.max(1)
            mask_u = (max_prob >= self.conf_thre).float()
            self.num_samples += mask_u[start_idx:].sum()
            self.total_num += label_u[start_idx:].size(0)
            correct_ps_label = (label_u[start_idx:] == label_u_gt).float()
            self.label_ps_quality += correct_ps_label.sum()
            r = np.random.rand(1)
            if r < self.cutmix_prob and not self.finetune:
                input_u2, label_u_mix, label_mix_weight, mask_u_mix = cutmix(
                         input_u2, label_u, self.cutmix_beta, mask_u)
            else:
                label_u_mix, mask_u_mix = label_u, mask_u
                label_mix_weight = 1.
        use_mixstyle = False
        output_u = self.model_inference(input_u2)
        assert label_mix_weight == 1
        loss_u = F.cross_entropy(output_u, label_u, reduction='none')
        loss_u = label_mix_weight * (loss_u * mask_u).mean()
        loss_u_mix = F.cross_entropy(output_u, label_u_mix, reduction='none')
        loss_u_mix = (1. - label_mix_weight) * (loss_u_mix * mask_u_mix).mean()
        loss_summary = {
            'loss_u': loss_u.item(),
            'acc_u': compute_accuracy(output_u, label_u)[0].item(),
            'loss_u_mix': loss_u_mix.item(),
            'threshold': self.conf_thre,
            'threshold_batch': threshold.item()
        }

        loss = (loss_u + loss_u_mix) * self.weight_u
        self.model_backward_and_update(loss)

        if self.lr_sched_type != 'fixmatch':
            if (self.batch_idx + 1) == self.num_batches:
                self.update_lr()
        else:
            self.update_lr()

        return loss_summary

    def ent(self, output):
        return - torch.mean(output * torch.log(output + 1e-6))

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
    def test(self, print_sample_num=True):
        """A generic testing pipeline."""
        if print_sample_num:
            # display samples above the threshold
            print('samples above the threshold {}({}/{}), correct pseudo labels {}({}/{})'.format(
                float(self.num_samples) / self.total_num,
                self.num_samples,
                self.total_num,
                float(self.label_ps_quality) / self.total_num,
                self.label_ps_quality,
                self.total_num))
            if float(self.num_samples) / self.total_num > self.retrain_ratio:
                self.high_ratio_tar = True
        self.label_ps_quality = 0
        self.num_samples = 0
        self.total_num = 0
        self.set_model_mode('eval')
        self.F_labeler.eval()
        self.C_labeler.eval()
        self.evaluator.reset()
        self.evaluator_new_model.reset()

        split = self.cfg.TEST.SPLIT
        print('Do evaluation on {} set'.format(split))
        data_loader = self.val_loader if split == 'val' else self.test_loader
        assert data_loader is not None

        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            output = self.labeler_inference(input)
            self.evaluator.process(output, label)
            if self.finetune:
                pred = self.model_inference(input)
                self.evaluator_new_model.process(pred, label)

        results = self.evaluator.evaluate()
        if self.finetune:
            results_new_model = self.evaluator_new_model.evaluate()
            self.acc_new_model.append(results_new_model['accuracy'])

        for k, v in results.items():
            tag = '{}/{}'.format(split, k)
            self.write_scalar(tag, v, self.epoch)
        self.acc.append(results['accuracy'])

        print('Until epoch {}, best accuracy of student model {}'.format(self.epoch, max(self.acc)))
        if len(self.acc_new_model) > 0:
            print('  best accuracy of new target-specific model {}'.format(max(self.acc_new_model)))

    def parse_batch_train(self, batch_x, batch_u):
        input_x = batch_x['img']
        input_x2 = batch_x['img2']
        label_x = batch_x['label']
        input_u = batch_u['img']
        input_u2 = batch_u['img2']
        label_u = batch_u['label']

        input_x = input_x.to(self.device)
        input_x2 = input_x2.to(self.device)
        label_x = label_x.to(self.device)
        input_u = input_u.to(self.device)
        input_u2 = input_u2.to(self.device)
        label_u = label_u.to(self.device)
        return input_x, input_x2, label_x, input_u, input_u2, label_u

    def model_inference(self, input):
        f = self.model(input)
        if self.method == 'MCD':
            z = self.C(f)
            p = F.softmax(z, 1)
        elif self.method == 'M3SDA':
            p = 0
            for C_i in self.C:
                z = C_i(f)
                if isinstance(z, tuple):
                    z = z[0]
                p += F.softmax(z, 1)
            p = p / len(self.C)
        return p

    def labeler_inference(self, input):
        f = self.F_labeler(input)
        if self.method == 'MCD':
            z = self.C_labeler(f)
            p = F.softmax(z, 1)
        elif self.method == 'M3SDA':
            p = 0
            for C_i in self.C_labeler:
                z = C_i(f)
                p += F.softmax(z, 1)
            p = p / len(self.C_labeler)
        return p
