"""
Modified from https://github.com/xternalz/WideResNet-pytorch
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .build import BACKBONE_REGISTRY
from .backbone import Backbone


class GlobalShiftV2Portion(nn.Module):

    def __init__(self, scale=1, portion=0.5):
        super(GlobalShiftV2Portion, self).__init__()
        self.scale = scale
        self.portion = portion
        self.index = (
            torch.arange(0, scale * scale).view(1, scale * scale) +
            torch.arange(0, scale * scale).view(scale * scale, 1)).fmod(
                scale * scale).long()

    def shift_scale(self, x):
        b, c, h, w = x.shape

        x = x.view(b, self.scale * self.scale, c // (self.scale * self.scale),
                   self.scale, h // self.scale, self.scale,
                   w // self.scale)  # dim=7
        x = x.permute(0, 1, 2, 4, 6, 3,
                      5).contiguous().view(b, self.scale * self.scale,
                                           c // (self.scale * self.scale),
                                           h // self.scale, w // self.scale,
                                           self.scale * self.scale)  # dim=6
        index = self.index.view(1, self.scale * self.scale, 1, 1, 1,
                                self.scale * self.scale).repeat(
                                    b, 1, c // (self.scale * self.scale),
                                    h // self.scale, w // self.scale,
                                    1).to(x.device)  # dim=6
        x = torch.gather(x, dim=5, index=index)
        x = x.view(b, c, h // self.scale, w // self.scale, self.scale,
                   self.scale).permute(0, 1, 4, 2, 5,
                                       3).contiguous().view(b, c, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape
        if self.portion > 0:
            shift = int(self.portion * c)
            if math.floor(shift /
                          (self.scale * self.scale)) != shift / (self.scale *
                                                                 self.scale):
                shift = math.floor(shift / (self.scale * self.scale)) * (
                    self.scale * self.scale)
            keep_x, shift_x = torch.split(
                x, split_size_or_sections=[c - shift, shift], dim=1)
            shift_x = self.shift_scale(shift_x)
            x = torch.cat([keep_x, shift_x], dim=1)

        return x

class GRCBasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.LeakyReLU(0.01, inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.LeakyReLU(0.01, inplace=True)
        self.conv2 = nn.Conv2d(
            out_planes,
            out_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=False
        ) or None
        self.shift = GlobalShiftV2Portion(4, 0.5)

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
            x = self.shift(x)
        else:
            out = self.relu1(self.bn1(x))
            out = self.shift(out)
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
       
        # out = self.shift(out)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)
    
    
class BasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.LeakyReLU(0.01, inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.LeakyReLU(0.01, inplace=True)
        self.conv2 = nn.Conv2d(
            out_planes,
            out_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=False
        ) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):

    def __init__(
        self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0
    ):
        super().__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, dropRate
        )

    def _make_layer(
        self, block, in_planes, out_planes, nb_layers, stride, dropRate
    ):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(
                block(
                    i == 0 and in_planes or out_planes, out_planes,
                    i == 0 and stride or 1, dropRate
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNetGRC(Backbone):

    def __init__(self, depth, widen_factor, dropRate=0.0):
        super().__init__()
        nChannels = [
            16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor
        ]
        self.shift = GlobalShiftV2Portion(4, 0.5)
        assert ((depth-4) % 6 == 0)
        n = (depth-4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(
            3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        # 1st block
        self.block1 = NetworkBlock(
            n, nChannels[0], nChannels[1], GRCBasicBlock, 1, dropRate
        )
        # 2nd block
        self.block2 = NetworkBlock(
            n, nChannels[1], nChannels[2], block, 2, dropRate
        )
        # 3rd block
        self.block3 = NetworkBlock(
            n, nChannels[2], nChannels[3], block, 2, dropRate
        )
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.LeakyReLU(0.01, inplace=True)

        self._out_features = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        # x = self.shift(x)
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        return out.view(out.size(0), -1)


@BACKBONE_REGISTRY.register()
def wide_resnet_grc_28_2(**kwargs):
    return WideResNetGRC(28, 2)
