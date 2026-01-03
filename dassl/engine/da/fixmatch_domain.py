import copy
import torch
from torch.nn import functional as F

from dassl.data import DataManager
from dassl.engine import TRAINER_REGISTRY, TrainerXU
from dassl.metrics import compute_accuracy
from dassl.data.transforms import build_transform
from dassl.engine.ssl import FixMatch
from dassl.evaluation import build_evaluator
from dassl.engine.trainer import SimpleNet
from dassl.utils import count_num_param
from dassl.optim import build_optimizer, build_lr_scheduler

class DomainBasedNet(SimpleNet):
    """A simple neural network composed of a CNN backbone
    and optionally a head such as mlp for classification.
    """

    def forward(self, x, return_feature=False, domain_btcsize=None, use_mixstyle=True):
        f = self.backbone(x, domain_btcsize=domain_btcsize, use_mixstyle=use_mixstyle)
        if self.head is not None:
            f = self.head(f)

        if self.classifier is None:
            return f

        y = self.classifier(f)

        if return_feature:
            return y, f

        return y
        
        
@TRAINER_REGISTRY.register()
class FixMatchDomain(FixMatch):
    """FixMatch: Simplifying Semi-Supervised Learning with
    Consistency and Confidence.

    https://arxiv.org/abs/2001.07685.

    FixMatch with Domain labels, sharing config file with base FixMatch
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
        
    def build_model(self):
        cfg = self.cfg
        print('Building model')
        self.model = DomainBasedNet(cfg, cfg.MODEL, self.num_classes)
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
        self.model.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.model)))
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model('model', self.model, self.optim, self.sched)

    
    def forward_backward(self, batch_x, batch_u):
        global_step = self.batch_idx + self.epoch * self.num_batches
        parsed_data = self.parse_batch_train(batch_x, batch_u)
        input_x, input_x2, label_x, input_u, input_u2 = parsed_data
        input_u1 = torch.cat([input_x, input_u], 0)
        input_u2 = torch.cat([input_x2, input_u2], 0)
        domain_btch_size = [self.split_batch for _ in range(self.n_domain)]

        # Generate artificial label
        with torch.no_grad():
            # when generating pseudo labels, we do not use MixStyle
            domain_btch_size.append(input_u.size(0))
            output_u = F.softmax(self.model(input_u1, use_mixstyle=False, domain_btcsize=None), 1)
            del domain_btch_size[-1]
            max_prob, label_u = output_u.max(1)
            mask_u = (max_prob >= self.conf_thre).float()
            self.num_samples += mask_u.sum()
            self.total_num += label_u.size()[0]
            values, indices = output_u.detach().topk(2, dim=1, largest=True, sorted=True)
        
        output_x = self.model(input_x, use_mixstyle=False)
        loss_x = F.cross_entropy(output_x, label_x)

        # Unsupervised loss
        if self.n_domain > 1:
            domain_btch_size.append(input_u2.size(0) - input_x2.size(0))
        else:
            domain_btch_size = None
        output_u = self.model(input_u2, domain_btcsize=domain_btch_size, use_mixstyle=True)
        loss_u = F.cross_entropy(output_u, label_u, reduction='none')
        loss_u = (loss_u * mask_u).mean()
  
        label_mix = False
        if label_mix:
            low_conf = ~mask_u.bool().unsqueeze(1)
            
            warmup = self.epoch / 5.
            loss_u_mix = values[:, 0] * F.cross_entropy(output_u, indices[:, 0], reduction='none') + \
                         values[:, 1] * F.cross_entropy(output_u, indices[:, 1], reduction='none')
            loss_u_mix = (loss_u_mix * low_conf * warmup).mean()
            loss_u += loss_u_mix

        loss = loss_x + (loss_u) * self.weight_u
        self.model_backward_and_update(loss)

        ema_alpha = min(1 - 1 / (global_step+1), self.ema_alpha)
        self.ema_model_update(self.model, self.teacher, ema_alpha)

        loss_summary = {
            'loss_x': loss_x.item(),
            'acc_x': compute_accuracy(output_x, label_x)[0].item(),
            'loss_u': loss_u.item(),
            'acc_u': compute_accuracy(output_u, label_u)[0].item(),
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
            output = self.model_inference(input)
            self.evaluator.process(output, label)
            output_teacher = self.teacher(input)
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


