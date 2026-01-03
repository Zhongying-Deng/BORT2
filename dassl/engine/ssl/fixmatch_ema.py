import copy
import numpy as np
import torch
from torch.nn import functional as F

from dassl.data import DataManager
from dassl.engine import TRAINER_REGISTRY, TrainerXU
from dassl.metrics import compute_accuracy
from dassl.data.transforms import build_transform
from dassl.engine.ssl import FixMatch
from dassl.evaluation import build_evaluator


@TRAINER_REGISTRY.register()
class FixMatchEMA(FixMatch):
    """FixMatch: Simplifying Semi-Supervised Learning with
    Consistency and Confidence.

    https://arxiv.org/abs/2001.07685.

    FixMatch with EMA, sharing config file with base FixMatch
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
        self.correct_high_conf = 0
        self.label_ps_quality = 0
        self.lr_sched_type = cfg.OPTIM.LR_SCHEDULER

    def forward_backward(self, batch_x, batch_u):
        global_step = self.batch_idx + self.epoch * self.num_batches
        #global_step = self.epoch + self.batch_idx / self.num_batches
        parsed_data = self.parse_batch_train(batch_x, batch_u)
        input_x, input_x2, label_x, _, input_u, input_u2, label_u_gt = parsed_data
        input_u = torch.cat([input_x, input_u], 0)
        input_u2 = torch.cat([input_x2, input_u2], 0)

        bs_src = input_x.size(0)
        # Generate artificial label
        with torch.no_grad():
            output_u = F.softmax(self.model(input_u), 1)
            max_prob, label_u = output_u.max(1)
            mask_u = (max_prob >= self.conf_thre).float()
            #img_shape = input_u.shape
            #img_noise = torch.randn(img_shape[0], img_shape[1], img_shape[2], img_shape[3]) * 0.15
            #input_u3 = torch.flip(input_u, [3]) + img_noise.to(self.device)
            #output_u3 = F.softmax(self.model(input_u3), 1)
            #max_prob3, label_u3 = output_u3.max(1)
            #mask_u3 = (max_prob3 >= self.conf_thre).float()
            #feat_teacher = self.teacher(input_u)
            #output_u_teacher = F.softmax(feat_teacher, 1)
            #max_prob_t, label_u_t = output_u_teacher.max(1)
            #mask_u_t = (max_prob_t >= self.conf_thre).float()
            ## pseudo label of EMA and student model should be consistent
            #label_consist = (label_u == label_u3).float()
            ## accept the pseudo label with a probility
            #avg_prob = (max_prob_t + max_prob)/2.
            #probility = torch.tensor(np.random.rand(max_prob_t.size(0)))
            #random_mask = (avg_prob > probility.float().to(self.device)).float()
            #mask_u = mask_u * mask_u_t * label_consist * random_mask
            ##mask_u = mask_u * mask_u3 * label_consist

            self.num_samples += mask_u[bs_src:].sum()
            self.total_num += label_u[bs_src:].size()[0]
            correct_ds_label = (label_u[bs_src:] == label_u_gt).float()
            self.correct_high_conf += (mask_u[bs_src:] * correct_ds_label).sum()
            self.label_ps_quality += correct_ds_label.sum()
        # Supervised loss
        output_x = self.model(input_x)
        loss_x = F.cross_entropy(output_x, label_x)

        # Unsupervised loss
        output_u = self.model(input_u2)
        loss_u = F.cross_entropy(output_u, label_u, reduction='none')
        loss_u = (loss_u * mask_u).mean()

        loss = loss_x + loss_u * self.weight_u
        self.model_backward_and_update(loss)

        ema_alpha = min(1 - 1 / (global_step+1), self.ema_alpha)
        self.ema_model_update(self.model, self.teacher, ema_alpha)

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
        self.total_num += 1e-7
        print('samples above the threshold {}({}/{}), among them correct labels'
            ' {} (rate {}), correct predictions {}'.format(
            float(self.num_samples ) / self.total_num, self.num_samples, self.total_num,
            self.correct_high_conf, float(self.correct_high_conf) / float(self.num_samples + 1e-7),
            float(self.label_ps_quality) / self.total_num)
            )
        self.num_samples = 0
        self.total_num = 0
        self.correct_high_conf = 0
        self.label_ps_quality = 0

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
        if len(self.acc) > 1:
            if results['accuracy'] >= max(self.acc):
                # Save model
                self.save_model(self.epoch, self.output_dir, is_best=True)
        print('Until epoch {}, best accuracy of student model {}, teacher model {}'.format(self.epoch, max(self.acc), max(self.acc_teacher)))

    def parse_batch_train(self, batch_x, batch_u):
        input_x = batch_x['img']
        input_x2 = batch_x['img2']
        label_x = batch_x['label']
        domain_x = batch_x['domain']
        input_u = batch_u['img']
        input_u2 = batch_u['img2']
        label_u = batch_u['label']
        #label_x = create_onehot(label_x, self.num_classes)

        input_x = input_x.to(self.device)
        input_x2 = input_x2.to(self.device)
        label_x = label_x.to(self.device)
        input_u = input_u.to(self.device)
        input_u2 = input_u2.to(self.device)
        label_u = label_u.to(self.device)
        return input_x, input_x2, label_x, domain_x, input_u, input_u2, label_u

