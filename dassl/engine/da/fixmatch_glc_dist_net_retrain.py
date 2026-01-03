import copy
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional as F
from torch.distributions import Normal
from torch.distributions.multivariate_normal import MultivariateNormal

from dassl.engine import TRAINER_REGISTRY
from dassl.metrics import compute_accuracy
from dassl.engine.ssl import FixMatch
from dassl.evaluation import build_evaluator
from dassl.engine.trainer import SimpleNet
from dassl.utils import count_num_param, load_pretrained_weights
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.modeling import build_head, build_backbone


def domain_discrepancy(out1, out2, loss_type):
    def huber_loss(e, d=1):
        t = torch.abs(e)
        ret = torch.where(t < d, 0.5 * t ** 2, d * (t - 0.5 * d))
        return torch.mean(ret)

    diff = out1 - out2
    if loss_type == 'L1':
        loss = torch.mean(torch.abs(diff))
    elif loss_type == 'Huber':
        loss = huber_loss(diff)
    else:
        loss = torch.mean(diff * diff)
    return loss


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
        use_dist_net = False
        for k, v in kwargs.items():
            if 'use_dist_net' in k:
                use_dist_net = v
        model_name = model_cfg.BACKBONE.NAME
        if use_dist_net:
            model_name = model_name + '_dist_net'
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
        sampled_feat = []  # None
        if isinstance(f, tuple):
            f, sig = f
            if self.dist_net and self.training:
                loss = 0.
                sigma_avg = 5
                threshold = 4. 
                sample_num = 1
                for _ in range(sample_num):
                    z = []
                    for ind in range(f.shape[0]):
                        scale_diag = F.softplus(sig[ind, :])
                        
                        entropy = torch.mean(torch.log(scale_diag + 1e-7))
                        loss += F.relu(threshold - entropy)
                        noise = Normal(0, 1).sample(scale_diag.size())
                        if scale_diag.is_cuda:
                            noise = noise.cuda()
                        sampled_dist = scale_diag * noise + f[ind, :]
                        z.append(sampled_dist.unsqueeze(0))
                    sampled_feat.append(torch.cat(z, 0))
                loss /= (f.shape[0] * sample_num)

        if self.head is not None:
            f = self.head(f)
            if len(sampled_feat) > 0:
                for ind, feat in enumerate(sampled_feat):
                    sampled_feat[ind] = self.head(feat)

        if self.classifier is None:
            return f

        y = self.classifier(f)
        if len(sampled_feat) > 0:
            y_sample = []
            for feat in sampled_feat:
                y_sample.append(self.classifier(feat))

        if return_feature:
            if len(sampled_feat) > 0:
                return y, y_sample, loss, f, F.softplus(sig)
            else:
                return y, f

        if len(sampled_feat) > 0:
            return y, y_sample, loss
        else:
            return y


@TRAINER_REGISTRY.register()
class FixMatchGLCDistNetRetrain(FixMatch):
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
            self.thre_param = - torch.log(1 / torch.tensor(self.conf_thre) - 1)
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
        self.finetune = True
        self.weight_sampling = 1.
        self.weight_ent = 0.1
        self.probs = []
        self.labels = []
        self.masks = []

    def build_model(self):
        cfg = self.cfg
        print('Building model')
        ms_cfg = cfg.TRAINER.MIXSTYLE.CONFIG
        self.model = DomainBasedNet(cfg, cfg.MODEL, self.num_classes, ms_cfg=ms_cfg, use_dist_net=False)
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
            # when using DAC-Net as 1st step training
            cls_weight = cfg.MODEL.INIT_WEIGHTS
            cls_weight = cls_weight.replace('model', 'classifier', 1)
            import os
            if os.path.exists(cls_weight):
                load_pretrained_weights(self.model.classifier, cls_weight)
            else:
                print(cls_weight)
        self.model.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.model)))
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model('model', self.model, self.optim, self.sched)

    def forward_backward(self, batch_x, batch_u):
        global_step = self.batch_idx + self.epoch * self.num_batches
        ema_alpha = min(1 - 1 / (global_step + 1), self.ema_alpha)

        lmda = None
        parsed_data = self.parse_batch_train(batch_x, batch_u)
        input_x, input_x2, label_x, input_u, input_u2, label_u_gt = parsed_data

        if self.finetune:
            if not self.built_new_optim:
                cfg = self.cfg
                ms_cfg = self.cfg.TRAINER.MIXSTYLE.CONFIG
                self.new_model = DomainBasedNet(self.cfg, self.cfg.MODEL,
                                                self.num_classes, ms_cfg=ms_cfg, use_dist_net=True)
                self.new_model.to(self.device)
                load_pretrained_weights(self.new_model, cfg.MODEL.INIT_WEIGHTS)
                self.optim_new_model = build_optimizer(self.new_model, self.cfg.OPTIM, lr_mult=0.1)
                self.sched_new_model = build_lr_scheduler(self.optim_new_model, self.cfg.OPTIM)
                self.register_model('model_tar', self.new_model,
                                    self.optim_new_model, self.sched_new_model)
                self.built_new_optim = True
                self.test(False)
                self.new_model.train()
                del self._optims['model']
                del self._models['model']
                del self._scheds['model']

            self.model.eval()
            
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
                    threshold = entropy / avg_ent
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

        if (self.epoch == 0):
            # this epoch is used for updating C_hat_transpose
            hyper_step = False

            self.probs.extend(list(output_u.data.cpu().numpy()))
            mask_u = (max_prob >= self.thre_param).float()
            self.labels.extend(list(label_u.data.cpu().numpy()))
            self.masks.extend(list(mask_u.data.cpu().numpy()))
            loss_summary = {
                'loss_u': torch.tensor(0.).item()
            }
        else:
            # updating C_hat_transpose
            probs = np.array(self.probs, dtype=np.float32)
            masks = np.array(self.masks, dtype=np.float32)
            C_hat = np.zeros((self.num_classes, self.num_classes))
            for label in range(self.num_classes):
                indices = np.arange(len(self.labels))[
                    np.isclose(np.array(self.labels), label)]

                C_hat[label] = np.mean(np.expand_dims(masks[indices], 1) * probs[indices], axis=0, keepdims=True)
            C_hat_transpose = torch.from_numpy(C_hat.T.astype(np.float32))
            C_hat_transpose = Variable(C_hat_transpose.cuda(), requires_grad=False)

            use_mixstyle = False
            output = self.new_model(
                input_u2,
                use_mixstyle=use_mixstyle,
                domain_btcsize=None,
                lmda=None,
                return_feature=True
            )
            output_u, output_u_sample, loss_entropy, feat, sig = output
            loss_u = F.cross_entropy(output_u, label_u, reduction='none')
            loss_u = (loss_u * mask_u).sum() / (mask_u.sum() + 1e-5)

            pre1 = C_hat_transpose[torch.cuda.LongTensor(label_u.data)]
            pred_u = F.softmax(output_u, 1)
            pre2 = torch.mul(pred_u, pre1)
            loss_u_mix = -(torch.log(pre2.sum(1)+1e-5)) * (1 - mask_u)
            loss_u_mix = loss_u_mix.sum() / ((1 - mask_u).sum() + 1e-5)
            loss_summary = {
                'loss_u': loss_u.item(),
                'acc_u': compute_accuracy(output_u, label_u)[0].item(),
                'loss_u_mix': loss_u_mix.item(),
                'threshold': self.conf_thre
            }

            loss_sampled = 0.
            for logit in output_u_sample:
                loss_s = F.cross_entropy(logit, label_u, reduction='none')
                loss_sampled += (loss_s * mask_u).mean()
            loss_sampled /= len(output_u_sample)

            loss = loss_sampled + self.weight_ent * loss_entropy + loss_u + loss_u_mix
            loss_summary['loss_sampled'] = self.weight_sampling * loss_sampled.item()
            loss_summary['loss_entropy'] = self.weight_ent * loss_entropy.item()

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

    def cross_entropy_bp_to_label(self, input, target, reduction="mean"):
        exp = torch.exp(input)
        tmp1 = exp.gather(1, target.unsqueeze(-1)).squeeze()
        tmp2 = exp.sum(1)
        softmax = tmp1 / tmp2
        log = -torch.log(softmax + 1e-10)
        if reduction == "mean":
            return log.mean()
        elif reduction == "sum":
            return log.sum()
        else:
            return log

    def cross_entropy_bp_to_onehot(self, fc_out, label, reduction="mean"):
        loss = label * torch.softmax(fc_out, 1)
        log = torch.log(torch.sum(loss, 1) + 1e-10)
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
        print('Until epoch {}, best accuracy of student model {}, teacher model {}'.format(self.epoch, max(self.acc),
                                                                                           max(self.acc_teacher)))
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
