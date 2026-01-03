import copy
from random import uniform
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Normal
import torchvision.transforms as T

from dassl.data import DataManager
from dassl.engine import TRAINER_REGISTRY, TrainerXU
from dassl.metrics import compute_accuracy
from dassl.data.transforms import build_transform
from dassl.engine.ssl import FixMatch
from dassl.evaluation import build_evaluator
from dassl.engine.trainer import SimpleNet
from dassl.utils import count_num_param, load_pretrained_weights, cutmix, mix_amplitude, MetaOptimizer
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

    def forward(self, x):
        z1 = self.c1(x)
        return z1


class DomainBasedNet(SimpleNet):
    """A simple neural network composed of a CNN backbone
    and optionally a head such as mlp for classification.
    """
    def __init__(self, cfg, model_cfg, num_classes, **kwargs):
        super().__init__(cfg, model_cfg, num_classes)
        use_dist_net = False
        for k, v in kwargs.items():
            if 'use_dist_net' in k:
                use_dist_net=v
        model_name = model_cfg.BACKBONE.NAME
        if use_dist_net:
            if 'drt' in model_name:
                model_name = model_name + '_dist_net'
            else:
                model_name = model_name + '_ms_dist_net'
            print('use DistributionNet {}.'.format(model_name))
        self.backbone = build_backbone(
                model_name,
                verbose=cfg.VERBOSE,
                pretrained=model_cfg.BACKBONE.PRETRAINED,
                **kwargs
                )
        fdim = self.backbone.out_features
        if 'ms' in model_cfg.BACKBONE.NAME:
            self.use_mixstyle = True
        else:
            self.use_mixstyle = False
        if 'dist_net' in model_name:
            self.dist_net = True
        else:
            self.dist_net = False
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
        sampled_feat = None
        if isinstance(f, tuple):
            f, sig = f
            if self.dist_net and self.training:
                z = []
                loss = 0.
                sigma_avg = 5
                threshold = 4. 
                for ind in range(f.shape[0]):
                    scale_diag = F.softplus(sig[ind, :])
                    
                    entropy = torch.mean(torch.log(scale_diag + 1e-7))
                    loss += F.relu(threshold - entropy)
                    noise = Normal(0, 1).sample(scale_diag.size())
                    if scale_diag.is_cuda:
                        noise = noise.cuda()
                    sampled_dist = scale_diag * noise + f[ind, :]
                    z.append(sampled_dist.unsqueeze(0))
                sampled_feat = torch.cat(z, 0)
                loss /= f.shape[0]

        if self.head is not None:
            f = self.head(f)
            if sampled_feat is not None:
                sampled_feat = self.head(sampled_feat)

        if self.classifier is None:
            if sampled_feat is not None:
                return f, sampled_feat, loss, f, F.softplus(sig)
            else:
                return f

        y = self.classifier(f)
        if sampled_feat is not None:
            y_sample = self.classifier(sampled_feat)

        if return_feature:
            if sampled_feat is not None:
                return y, y_sample, loss, f, F.softplus(sig)
            else:
                return y, f
    
        if sampled_feat is not None:
            return y, y_sample, loss
        else:
            return y
            

@TRAINER_REGISTRY.register()
class FixMatchBaselinesDistNetMetaLearnRetrain(FixMatch):
    """BORT2: Zhongying Deng, Da Li, Xiaojiang Peng, Yi-Zhe Song, Tao Xiang. "BORT2: Bi-level optimization for robust target training in multi-source domain adaptation." Pattern Recognition (2026)](https://doi.org/10.1016/j.patcog.2025.112367)
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
        self.weight_sampling = 0.1
        self.weight_ent = 0.1

    def build_model(self):
        self.method = 'M3SDA'
        cfg = self.cfg
        print('Building model for {}'.format(self.method))
        ms_cfg = cfg.TRAINER.MIXSTYLE.CONFIG
        if self.method == 'DANN':
            self.model = DomainBasedNet(
                    cfg, cfg.MODEL, 
                    self.num_classes, 
                    ms_cfg=ms_cfg, 
                    use_dist_net=True
                    )
        else:
            self.model = DomainBasedNet(cfg, cfg.MODEL, 0, ms_cfg=ms_cfg, use_dist_net=True)
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
        else:
            assert False, 'MODEL.INIT_WEIGHTS is empty. Initial weights not provided.'
        fdim = self.model.fdim
        self.model.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.model)))
        self.optim = build_optimizer(self.model, cfg.OPTIM, lr_mult=0.1)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model('model', self.model, self.optim, self.sched)
        if self.method != 'DANN':
            if self.method == 'MCD':
                print('Building C')
                self.C = nn.Linear(fdim, self.num_classes)
                if cfg.MODEL.INIT_WEIGHTS:
                    weights_c = cfg.MODEL.INIT_WEIGHTS
                    weights_c = weights_c.replace('model', 'C', 1)
            elif self.method == 'M3SDA':
                self.C = nn.ModuleList(
                        [
                            PairClassifiers(fdim, self.num_classes)
                            for _ in range(self.dm.num_source_domains)
                            ]
                        )
                if cfg.MODEL.INIT_WEIGHTS:
                    weights_c = cfg.MODEL.INIT_WEIGHTS
                    weights_c = weights_c.replace('model', 'C', 1)
            else:
                pass
            load_pretrained_weights(self.C, weights_c)
            self.C.to(self.device)
            
            print('# params: {:,}'.format(count_num_param(self.C)))
            self.optim_C = build_optimizer(self.C, cfg.OPTIM, lr_mult=0.1)
            self.sched_C = build_lr_scheduler(self.optim_C, cfg.OPTIM)
            self.register_model('C', self.C, self.optim_C, self.sched_C)
        # load pseudo labeling function
        if self.method == 'DANN':
            self.F_labeler = SimpleNet(cfg, cfg.MODEL, self.num_classes)
        else:
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
            self.C_labeler.to(self.device)
            model_weights = copy.deepcopy(self.C.state_dict())
            self.C_labeler.load_state_dict(model_weights)
            self.C_labeler.eval()
        self.F_labeler.to(self.device)
        
        model_weights = copy.deepcopy(self.model.state_dict())
        load_pretrained_weights(self.F_labeler, cfg.MODEL.INIT_WEIGHTS)
        
        self.F_labeler.eval()

        lr = 0.00005 #cfg.OPTIM.LR
        self.meta_optim_type = cfg.TRAINER.METALEARN.TYPE
        self.meta_optim_lr = cfg.TRAINER.METALEARN.LR
        self.meta_optim_step = cfg.TRAINER.METALEARN.STEP
        self.meta_optim_truncate = cfg.TRAINER.METALEARN.TRUNCATE
        if self.meta_optim_type == 'adam':
            if self.method == 'DANN':
                optim_gen = torch.optim.Adam(self.F_labeler.parameters(), lr=lr)
            else:
                optim_gen = torch.optim.Adam([{"params": self.F_labeler.parameters()},
                        {"params": self.C_labeler.parameters()}], lr=lr)
        else:
            if self.method == 'DANN':
                optim_gen = torch.optim.SGD(self.F_labeler.parameters(), lr=lr)
            else:
                optim_gen = torch.optim.SGD([{"params": self.F_labeler.parameters()},
                        {"params": self.C_labeler.parameters()}], lr=lr)
        self.optim_gen = MetaOptimizer(optim_gen, hpo_lr=self.meta_optim_lr, 
                       truncate_iter=self.meta_optim_truncate, max_grad_norm=50)
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
        output = self.model_inference(input_u2)
        output_u, output_u_sample, loss_entropy, feat, sig = output
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

        loss_sampled = F.cross_entropy(output_u_sample, label_u, reduction='none')
        loss_sampled = (loss_sampled * mask_u).mean()
        loss = loss_sampled + self.weight_ent * loss_entropy
        loss_summary['loss_sampled'] = loss_sampled.item()
        loss_summary['loss_entropy'] = self.weight_ent * loss_entropy.item()
        self.model_backward_and_update(loss)

        if ((global_step+1) % self.meta_optim_step == 0):
            # do validation to update the style generator
            hyper_step = True
        else:
            hyper_step = False
        if hyper_step and self.finetune:
            self.F_labeler.train()
            if self.method != 'DANN':
                self.C_labeler.train()
            pred_u = self.labeler_inference(input_u1)
            max_prob, _ = F.softmax(pred_u, 1).max(1)
            label_u = F.gumbel_softmax(pred_u, tau=1., hard=True)
            mask_u = (max_prob >= self.conf_thre).float()
            output = self.model_inference(input_u2)
            output_u, output_u_sample, loss_entropy, feat, sig = output
            loss_sampled = 0.
            if not isinstance(output_u_sample, list):
                output_u_sample = [output_u_sample]
            for logit in output_u_sample:
                loss_s = self.cross_entropy_bp_to_onehot(logit, label_u, reduction='none')

                loss_sampled += (loss_s * mask_u).mean()
            loss_sampled /= len(output_u_sample)
            
            loss_train = loss_sampled + self.weight_ent * loss_entropy

            entropy_val = F.relu(torch.mean(torch.log(sig + 1e-7)))
            aux_params = list(self.F_labeler.parameters())
            if self.method != 'DANN':
                aux_params += list(self.C_labeler.parameters())
            params = list(self.model.parameters())
            if self.method != 'DANN':
                params += list(self.C.parameters())
            self.optim_gen.step(
                val_loss=entropy_val,
                train_loss=loss_train,
                aux_params=aux_params,
                parameters=params)
            loss_summary['loss_val'] = entropy_val.item()
            self.F_labeler.eval()
            if self.method != 'DANN':
                self.C_labeler.eval()
        
        if self.lr_sched_type != 'fixmatch':
            if (self.batch_idx + 1) == self.num_batches:
                self.update_lr()
        else:
            self.update_lr()

        return loss_summary

    def ent(self, output):
        return - torch.mean(output * torch.log(output + 1e-6))
    
    def cross_entropy_bp_to_onehot(self, fc_out, label, reduction="mean"):
        loss = label * torch.softmax(fc_out, 1)
        log = torch.log(torch.sum(loss, 1)+1e-10)
        if reduction == "mean": 
            return log.mean()
        elif reduction == "sum":
            return log.sum()
        else:
            return log

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
        if self.method != 'DANN':
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
        f = self.model(input, 
                    use_mixstyle=False,
                    domain_btcsize=None, 
                    lmda=None,
                    return_feature=True)
        if isinstance(f, tuple):
            if len(f) > 2:
                output, output_sample, loss_entropy, feat, sig = f
            else:
                output, _ = f
                output_sample = None
        else:
            output_sample = None
            output = f
        p_sample = None
        if self.method == 'MCD':
            z = self.C(output)
            p = F.softmax(z, 1)
            if output_sample is not None:
                z_sample = self.C(output_sample)
                p_sample = F.softmax(z_sample, 1)
        elif self.method == 'M3SDA':
            p = 0
            for C_i in self.C:
                z = C_i(output)
                if isinstance(z, tuple):
                    z = z[0]
                p += F.softmax(z, 1)
            p = p / len(self.C)
            if output_sample is not None:
                p_sample = 0
                for C_i in self.C:
                    z_sample = C_i(output_sample)
                    if isinstance(z_sample, tuple):
                        z_sample = z_sample[0]
                    p_sample += F.softmax(z_sample, 1)
                p_sample = p_sample / len(self.C)
        else:
            p = F.softmax(output, 1)
            if output_sample is not None:
                p_sample = F.softmax(output_sample, 1)

        if p_sample is not None:
            return p, p_sample, loss_entropy, feat, sig
        else:
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
                if isinstance(z, tuple):
                    z = z[0]
                p += F.softmax(z, 1)
            p = p / len(self.C_labeler)
        else:
            p = F.softmax(f, 1)
        return p
