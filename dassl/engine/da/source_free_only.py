import torch.nn as nn
from torch.nn import functional as F
import torch.nn.utils.weight_norm as WeightNorm

from dassl.engine import TRAINER_REGISTRY, TrainerXU
from dassl.metrics import compute_accuracy
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

        fdim = self.backbone.out_features

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
            weight_norm = True
            if weight_norm:
                self.classifier = WeightNorm(nn.Linear(fdim, num_classes))
            else:
                self.classifier = nn.Linear(fdim, num_classes)

        self._fdim = fdim

    @property
    def fdim(self):
        return self._fdim

    def forward(self, x, return_feature=False):
        f = self.backbone(x)
        if self.head is not None:
            f = self.head(f)

        if self.classifier is None:
            return f

        y = self.classifier(f)

        if return_feature:
            return y, f

        return y



@TRAINER_REGISTRY.register()
class SourceFreeOnly(TrainerXU):
    """Baseline model for domain adaptation, which is
    trained using source data only.
    """

    def forward_backward(self, batch_x, batch_u):
        input, label = self.parse_batch_train(batch_x, batch_u)
        output = self.model(input)
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
