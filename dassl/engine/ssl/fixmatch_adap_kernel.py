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


@TRAINER_REGISTRY.register()
class FixMatchAdapKernel(FixMatch):
    """FixMatch: Simplifying Semi-Supervised Learning with
    Consistency and Confidence.

    https://arxiv.org/abs/2001.07685.

    FixMatch for adaptive kernel (two classifiers for two domains) and 
       EMA(MeanTeacher https://arxiv.org/abs/1703.01780), 
       sharing config file with base FixMatch
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.weight_u = cfg.TRAINER.FIXMATCH.WEIGHT_U
        self.conf_thre = cfg.TRAINER.FIXMATCH.CONF_THRE
        self.ema_alpha = cfg.TRAINER.FIXMATCH.EMA_ALPHA
        # evaluator for EMA model 
        self.evaluator_teacher = build_evaluator(cfg, lab2cname=self.dm.lab2cname)
        self.evaluator_tar = build_evaluator(cfg, lab2cname=self.dm.lab2cname)
        
        self.acc = []
        self.acc_tar = []
        self.acc_teacher = []
        self.num_samples = 0
        self.total_num = 0

    def build_model(self):
        cfg = self.cfg
        print('Building model')
        self.model = SimpleNet(cfg, cfg.MODEL, 0)
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
        self.model.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.model)))
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model('model', self.model, self.optim, self.sched)
        
        fdim = self.model.fdim
        self.classifier = torch.nn.Linear(fdim, self.num_classes)
        self.classifier.to(self.device)
        self.optim_c = build_optimizer(self.classifier, cfg.OPTIM)
        self.sched_c = build_lr_scheduler(self.optim_c, cfg.OPTIM)
        self.register_model('classifier', self.classifier, self.optim_c, self.sched_c)
        print('Building the classifier for the target domain')
        self.classifier_tar = torch.nn.Linear(fdim, self.num_classes)
        self.classifier_tar.to(self.device)
        self.optim_tar = build_optimizer(self.classifier_tar, cfg.OPTIM)
        self.sched_tar = build_lr_scheduler(self.optim_tar, cfg.OPTIM)
        self.register_model('classifier_tar', self.classifier_tar, self.optim_tar, self.sched_tar)

        # build EMA model
        self.teacher = copy.deepcopy(self.model)
        self.teacher.train()
        for param in self.teacher.parameters():
            param.requires_grad_(False)

        self.teacher_classifier = copy.deepcopy(self.classifier)
        self.teacher_classifier.train()
        for param in self.teacher_classifier.parameters():
            param.requires_grad_(False)

    
    def forward_backward(self, batch_x, batch_u):
        global_step = self.batch_idx + self.epoch * self.num_batches
        #global_step = self.epoch + self.batch_idx / self.num_batches
        parsed_data = self.parse_batch_train(batch_x, batch_u)
        input_x, input_x2, label_x, input_u, input_u2 = parsed_data
        input_u = torch.cat([input_x, input_u], 0)
        input_u2 = torch.cat([input_x2, input_u2], 0)

        # Generate artificial label
        feat_weak = self.model(input_u)
        with torch.no_grad():
            output_u = F.softmax(self.classifier(feat_weak), 1)
            max_prob, label_u = output_u.max(1)
            mask_u = (max_prob >= self.conf_thre).float()
            self.num_samples += mask_u.sum()
            self.total_num += label_u.size()[0]

        # Supervised loss
        feat_x = self.model(input_x)
        output_x = self.classifier(feat_x)
        loss_x = F.cross_entropy(output_x, label_x)

        # Unsupervised loss
        feat_u = self.model(input_u2)
        output_u = self.classifier_tar(feat_u)
        loss_u = F.cross_entropy(output_u, label_u, reduction='none')
        loss_u = (loss_u * mask_u).mean()

        loss = loss_x + loss_u * self.weight_u
        self.model_backward_and_update(loss)
      
        # update EMA model
        ema_alpha = min(1 - 1 / (global_step+1), self.ema_alpha)
        self.ema_model_update(self.model, self.teacher, ema_alpha)
        self.ema_model_update(self.classifier_tar, self.teacher_classifier, ema_alpha)

        loss_summary = {
            'loss_x': loss_x.item(),
            'acc_x': compute_accuracy(output_x, label_x)[0].item(),
            'loss_u': loss_u.item(),
            'acc_u': compute_accuracy(output_u, label_u)[0].item()
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
        #decay = min(1 - 1 / (step + 1), decay)
        with torch.no_grad():
            msd = model.state_dict()
            for k, ema_v in ema.state_dict().items():
                if needs_module:
                    k = 'module.' + k
                model_v = msd[k].detach()
                ema_v.copy_(ema_v * decay + (1. - decay) * model_v)
                # weight decay
                # if 'bn' not in k:
                #     msd[k] = msd[k] * (1. - self.wd)

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
        self.teacher_classifier.eval()
        self.evaluator.reset()
        self.evaluator_tar.reset()
        self.evaluator_teacher.reset()
        
        split = self.cfg.TEST.SPLIT
        print('Do evaluation on {} set'.format(split))
        data_loader = self.val_loader if split == 'val' else self.test_loader
        assert data_loader is not None
        
        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            feat = self.model_inference(input)
            output = self.classifier(feat)
            output_tar = self.classifier_tar(feat)
            self.evaluator.process(output, label)
            self.evaluator_tar.process(output_tar, label)
            output_teacher = self.teacher(input)
            output_teacher = self.teacher_classifier(output_teacher)
            self.evaluator_teacher.process(output_teacher, label)
            
        results = self.evaluator.evaluate()
        results_tar = self.evaluator_tar.evaluate()
        results_teacher = self.evaluator_teacher.evaluate()
            
        for k, v in results.items():
            tag = '{}/{}'.format(split, k)
            self.write_scalar(tag, v, self.epoch)
        self.acc.append(results['accuracy'])

        for k, v in results_tar.items():
            tag = 'tar_cls_{}/{}'.format(split, k)
            self.write_scalar(tag, v, self.epoch)
        self.acc_tar.append(results_tar['accuracy'])

        for k, v in results_teacher.items():
            tag_ema = 'ema_{}/{}'.format(split, k)
            self.write_scalar(tag_ema, v, self.epoch)
        self.teacher.train()
        self.teacher_classifier.train()
        self.acc_teacher.append(results_teacher['accuracy'])
        print('Until epoch {}, best accuracy of student model (src cls) {}, (tar cls) {}, teacher model (tar cls) {}'.format(self.epoch, max(self.acc), max(self.acc_tar), max(self.acc_teacher)))


