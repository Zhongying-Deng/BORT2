import copy
from random import uniform
import numpy as np
import torch
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

class DomainBasedNet(SimpleNet):
    """A simple neural network composed of a CNN backbone
    and optionally a head such as mlp for classification.
    """
    def __init__(self, cfg, model_cfg, num_classes, **kwargs):
        super().__init__(cfg, model_cfg, num_classes)
        self.backbone = build_backbone(
                model_cfg.BACKBONE.NAME,
                verbose=cfg.VERBOSE,
                pretrained=model_cfg.BACKBONE.PRETRAINED,
                #style_g_train=style_g_train,
                **kwargs
                )
        fdim = self.backbone.out_features
        if 'ms' in model_cfg.BACKBONE.NAME:
            self.use_mixstyle = True
        else:
            self.use_mixstyle = False
        self.head = None
        if model_cfg.HEAD.NAME and model_cfg.HEAD.HIDDEN_LAYERS:
            self.head = build_head(
                model_cfg.HEAD.NAME,
                verbose=cfg.VERBOSE,
                in_features=fdim,
                hidden_layers=model_cfg.HEAD.HIDDEN_LAYERS,
                activation=model_cfg.HEAD.ACTIVATION,
                bn=model_cfg.HEAD.BN,
                dropout=model_cfg.HEAD.DROPOUT,
            )
            fdim = self.head.out_features

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
class FixMatchMSCMRetrain(FixMatch):
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
        self.ent_thre = False
        if self.ent_thre:
            self.thre_param = - torch.log(1/torch.tensor(self.conf_thre) - 1)
        else:
            self.thre_param = cfg.TRAINER.FIXMATCH.CONF_THRE

        self.teacher = copy.deepcopy(self.model)
        self.teacher.train()
        for param in self.teacher.parameters():
            param.requires_grad_(False)
        self.evaluator_teacher = build_evaluator(cfg, lab2cname=self.dm.lab2cname)
        self.evaluator_new_model = build_evaluator(cfg, lab2cname=self.dm.lab2cname)
        self.acc = []
        self.acc_teacher = []
        self.acc_new_model = []
        self.num_samples = 0
        self.label_ps_quality = 0
        self.total_num = 1e-7
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
        self.finetune = False

    def build_model(self):
        cfg = self.cfg
        print('Building model')
        ms_cfg = cfg.TRAINER.MIXSTYLE.CONFIG
        self.model = DomainBasedNet(cfg, cfg.MODEL, self.num_classes, ms_cfg=ms_cfg, style_g_train=True)
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
        self.model.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.model)))
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model('model', self.model, self.optim, self.sched)

    def forward_backward(self, batch_x, batch_u):
        global_step = self.batch_idx + self.epoch * self.num_batches
        ema_alpha = min(1 - 1 / (global_step+1), self.ema_alpha)
        lmda = None
        parsed_data = self.parse_batch_train(batch_x, batch_u)
        input_x, input_x2, label_x, input_u, input_u2, label_u_gt = parsed_data
        if self.epoch > self.max_epoch - self.retrain_epoch or self.high_ratio_tar:
            self.finetune = True
        else:
            self.finetune = False
        if self.finetune:
            if not self.built_new_optim:
  
                ms_cfg = self.cfg.TRAINER.MIXSTYLE.CONFIG
                self.new_model = DomainBasedNet(self.cfg, self.cfg.MODEL, 
                     self.num_classes, ms_cfg=ms_cfg, style_g_train=True)
                self.new_model.to(self.device)
                model_weights = copy.deepcopy(self.model.state_dict())
                self.new_model.load_state_dict(model_weights)
                self.optim_new_model = build_optimizer(self.new_model, self.cfg.OPTIM, lr_mult=0.1)

                self.sched_new_model = build_lr_scheduler(self.optim_new_model, self.cfg.OPTIM)
                self.register_model('model_tar', self.new_model, 
                      self.optim_new_model, self.sched_new_model)
                self.built_new_optim = True
                self.test(False)
                del self._optims['model']
                del self._models['model']
                del self._scheds['model']
                
            self.model.eval()
            self.model.apply(fix_bn)
            for param in self.model.parameters():
                param.requires_grad_(False)
           
        if not self.finetune:
            input_u1 = torch.cat([input_x, input_u], 0)
            input_u2 = torch.cat([input_x2, input_u2], 0)
            start_idx = input_x.size(0)
        else:
            input_u1 = input_u
            start_idx = 0
        domain_btch_size = [self.split_batch for _ in range(self.n_domain)]

        use_mixstyle = False
        # Generate artificial label
        with torch.no_grad():
            # when generating pseudo labels, we do not use MixStyle
            domain_btch_size.append(input_u.size(0))
            output_u = F.softmax(self.model(input_u1, use_mixstyle=use_mixstyle,
                                        domain_btcsize=domain_btch_size, lmda=lmda), 1)
            del domain_btch_size[-1]
            if self.finetune:
                output_tmp = output_u[start_idx:]
                if self.ent_thre:
                    entropy = self.ent(output_tmp)
                    avg_ent = self.ent(torch.mean(output_tmp, dim=0))
                    threshold = entropy/avg_ent
                    self.thre_param = self.ema_alpha * self.thre_param + (1. - self.ema_alpha) * threshold
                    self.conf_thre = torch.sigmoid(self.thre_param)
                else:
                    prob_u, _ = output_tmp.max(1)
                    if self.batch_idx == 0 and (
                            self.epoch == self.max_epoch - self.retrain_epoch):
                        self.thre_param = torch.clamp(
                                prob_u.mean() + prob_u.std(),
                                0., self.thre_param
                        )
                        print('set the initial threshold to prob_u.mean() + prob_u.std(): {}'.format(self.thre_param))
                    threshold = prob_u.mean() - prob_u.std()
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
            
        if not self.finetune:
            output_x = self.model(input_x)
            loss_x = F.cross_entropy(output_x, label_x)

            # Unsupervised loss
            use_mixstyle = True
            domain_btch_size.append(input_u2.size(0) - input_x2.size(0))
            output_u = self.model(input_u2, use_mixstyle=use_mixstyle,
                              domain_btcsize=domain_btch_size, lmda=lmda)
            del domain_btch_size[-1]
        else:
            use_mixstyle = False
            output_u = self.new_model(input_u2, use_mixstyle=use_mixstyle,
                                  domain_btcsize=None, lmda=None)
            assert label_mix_weight == 1
        loss_u = F.cross_entropy(output_u, label_u, reduction='none')
        loss_u = label_mix_weight * (loss_u * mask_u).mean()
        loss_u_mix = F.cross_entropy(output_u, label_u_mix, reduction='none')
        loss_u_mix = (1. - label_mix_weight) * (loss_u_mix * mask_u_mix).mean()
        loss_summary = {
            'loss_u': loss_u.item(),
            'acc_u': compute_accuracy(output_u, label_u)[0].item(),
            'loss_u_mix': loss_u_mix.item(),
            'threshold': self.conf_thre
        }

        if not self.finetune:
            loss = loss_x + (loss_u + loss_u_mix) * self.weight_u
            loss_summary['loss_x'] = loss_x.item()
            loss_summary['acc_x'] = compute_accuracy(output_x, label_x)[0].item()
        else:
            loss = (loss_u + loss_u_mix) * self.weight_u
        self.model_backward_and_update(loss)
        
        if not self.finetune:
            self.ema_model_update(self.model, self.teacher, ema_alpha)

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
        self.teacher.eval()
        self.evaluator.reset()
        self.evaluator_teacher.reset()
        self.evaluator_new_model.reset()

        split = self.cfg.TEST.SPLIT
        print('Do evaluation on {} set'.format(split))
        data_loader = self.val_loader if split == 'val' else self.test_loader
        assert data_loader is not None

        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.evaluator.process(output, label)
            output_teacher = self.teacher(input)
            self.evaluator_teacher.process(output_teacher, label)
            if self.finetune:
                pred = self.new_model(input)
                self.evaluator_new_model.process(pred, label)

        results = self.evaluator.evaluate()
        results_teacher = self.evaluator_teacher.evaluate()
        if self.finetune:
            results_new_model = self.evaluator_new_model.evaluate()
            self.acc_new_model.append(results_new_model['accuracy'])

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


