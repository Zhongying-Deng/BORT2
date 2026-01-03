import torch.nn as nn
from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY, TrainerXU
from dassl.metrics import compute_accuracy
from dassl.utils import count_num_param
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.modeling import build_head, build_backbone


class SimpleNet(nn.Module):
    """A simple neural network composed of a CNN backbone
    and optionally a head such as mlp for classification.
    """

    def __init__(self, cfg, model_cfg, num_classes, **kwargs):
        super().__init__()
        self.backbone = build_backbone(
            model_cfg.BACKBONE.NAME,
            verbose=cfg.VERBOSE,
            pretrained=model_cfg.BACKBONE.PRETRAINED,
            **kwargs
        )
        self._fdim = self.backbone.out_features
    
    @property
    def fdim(self):
        return self._fdim

    def forward(self, x, return_feature=False):
        f = self.backbone(x)
        return f


class Head(nn.Module):
    """Head for VisDA dataset.
       stack of several Dropout-FC-BN-ReLU layers
    """

    def __init__(self, cfg, model_cfg, num_classes, fdim, **kwargs):
        super().__init__()
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
                **kwargs
            )
            fdim = self.head.out_features

        self.classifier = None
        if num_classes > 0:
            self.classifier = nn.Linear(fdim, num_classes)

        self._fdim = fdim

    @property
    def fdim(self):
        return self._fdim

    def forward(self, f):
        if self.head is not None:
            f = self.head(f)

        if self.classifier is None:
            return f

        y = self.classifier(f)
        return y


@TRAINER_REGISTRY.register()
class SourceOnlyVisDA(TrainerXU):
    """Baseline model for domain adaptation, which is
    trained using source data only.
    """
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
        self.classifier = Head(cfg, cfg.MODEL, self.num_classes, fdim)
        self.classifier.to(self.device)
        self.optim_c = build_optimizer(self.classifier, cfg.OPTIM, head_momentum=True)
        self.sched_c = build_lr_scheduler(self.optim_c, cfg.OPTIM)
        self.register_model('classifier', self.classifier, self.optim_c, self.sched_c)

    def forward_backward(self, batch_x, batch_u):
        input, label = self.parse_batch_train(batch_x, batch_u)
        output = self.classifier(self.model(input))
        loss = F.cross_entropy(output, label)
        self.model_backward_and_update(loss)

        loss_summary = {
            'loss': loss.item(),
            'acc': compute_accuracy(output, label)[0].item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        input = batch_x['img']
        label = batch_x['label']
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def model_inference(self, input):
        return self.classifier(self.model(input))

