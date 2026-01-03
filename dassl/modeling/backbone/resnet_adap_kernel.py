import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from .build import BACKBONE_REGISTRY
from .backbone import Backbone
from .dmc_layer import GlobalAdaptKernelModule, GlobalSpatialAdaptKernelModule, AdaptiveKernelModule, GlobalMultiScaleAdaptKernelModule, GlobalMultiScaleAdaptKernelBaseline, DIDAModule

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

reduction = 16

class ChannelGate(nn.Module):
    """A mini-network that generates channel-wise gates conditioned on input tensor."""

    def __init__(
        self,
        in_channels,
        num_gates=None,
        return_gates=False,
        gate_activation='relu',
        reduction=16,
        layer_norm=False
    ):
        super(ChannelGate, self).__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            in_channels,
            in_channels // reduction,
            kernel_size=1,
            bias=True,
            padding=0
        )
        self.norm1 = None
        if layer_norm:
            self.norm1 = nn.LayerNorm((in_channels // reduction, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            in_channels // reduction,
            num_gates,
            kernel_size=1,
            bias=True,
            padding=0
        )
        if gate_activation == 'sigmoid':
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == 'relu':
            self.gate_activation = nn.ReLU(inplace=True)
        elif gate_activation == 'linear':
            self.gate_activation = None
        else:
            raise RuntimeError(
                "Unknown gate activation: {}".format(gate_activation)
            )

    def forward(self, x):
        input = x
        x = self.global_avgpool(x)
        x = self.fc1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        if self.return_gates:
            return x
        return input * x


class IBN(nn.Module):
    r"""Instance-Batch Normalization layer from
    `"Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net"
    <https://arxiv.org/pdf/1807.09441.pdf>`
    Args:
        planes (int): Number of channels for the input tensor
        ratio (float): Ratio of instance normalization in the IBN layer
    """
    def __init__(self, planes, ratio=0.5):
        super(IBN, self).__init__()
        self.half = int(planes * ratio)
        self.IN = nn.InstanceNorm2d(self.half, affine=True)
        self.BN = nn.BatchNorm2d(planes - self.half)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class AdapKernelBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_ibn=False, shortcut=False):
        super().__init__()
        #self.conv1 = AdaptiveKernelModule(3, inplanes, planes, m=16, padding=None, stride=stride)
        #self.conv1 = GlobalAdaptKernelModule(3, inplanes, planes, m=16, padding=None, stride=stride)
        #self.conv1 = GlobalSpatialAdaptKernelModule(3, inplanes, planes, m=16, padding=None, stride=stride)
        #self.conv1 = GlobalMultiScaleAdaptKernelModule(3, inplanes, planes, m=reduction, padding=None, stride=stride)
        self.conv1 = DIDAModule(3, inplanes, planes, m=reduction, padding=None, stride=stride)
        #self.conv1 = conv3x3(inplanes, planes, stride)
        if use_ibn:
            self.bn1 = IBN(planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
        #self.bn1 = nn.InstanceNorm2d(planes, affine=True)
        self.relu = nn.ReLU(inplace=True)
        #self.conv2 = conv3x3(planes, planes)
        #self.conv2 = AdaptiveKernelModule(3, planes, planes, m=2, padding=None)
        #self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample if shortcut else None
        self.stride = stride
        self.shortcut = shortcut

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        #out = self.relu(out)

        #out = self.conv2(out)
        #out = self.bn2(out)

        if self.shortcut:
            if self.downsample is not None:
                residual = self.downsample(x)
            #print(x.size(), out.size())
            out += residual
        out = self.relu(out)

        return out


class AdapKernelBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_ibn=False, shortcut=False):
        super().__init__()
        #self.conv1 = AdaptiveKernelModule(3, inplanes, planes, m=2, padding=None, stride=stride)
        #self.conv1 = GlobalAdaptKernelModule(3, inplanes, planes, m=16, padding=None, stride=stride)
        #self.conv1 = GlobalMultiScaleAdaptKernelModule(3, inplanes, planes, m=reduction, padding=None, stride=stride)
        #self.conv1 = GlobalSpatialAdaptKernelModule(3, inplanes, planes, m=16, padding=None, stride=stride)
        self.conv1 = DIDAModule(3, inplanes, planes, m=reduction, padding=None, stride=stride)
        if use_ibn:
            self.bn1 = IBN(planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample if shortcut else None
        self.stride = stride
        self.shortcut = shortcut

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        #out = self.relu(out)

        #out = self.conv3(out)
        #out = self.bn3(out)
        if self.shortcut:
            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual
        out = self.relu(out)

        return out


class ResNet(Backbone):

    def __init__(self, block, adapblock, layers, **kwargs):
        self.inplanes = 64
        super().__init__()

        # backbone network
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        #self.conv1_adap = AdaptiveKernelModule(7, 3, 64, m=2, padding=3, stride=2)
        #self.bn1_adap = IBN(64)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.layer1_adap = self._make_adap_kernel_layer(adapblock, 64, layers[0], use_ibn=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        #self.layer_adap = adapblock(64, 64, use_ibn=True, shortcut=True)
        #self.layer2_adap = self._make_adap_kernel_layer(adapblock, 128, layers[1], stride=2, use_ibn=True)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        #self.layer_adap = adapblock(128, 128, use_ibn=True, shortcut=True)
        #self.layer3_adap = self._make_adap_kernel_layer(adapblock, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        #self.gate3 = ChannelGate(256)
        self.layer4_adap = self._make_adap_kernel_layer(adapblock, 512, layers[3], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        #self.gate4 = ChannelGate(512)
        #downsample = nn.Sequential(
        #        nn.Conv2d(
        #            64, 448,
        #            kernel_size=1,
        #            stride=1,
        #            bias=False
        #        ),
        #        nn.BatchNorm2d(448),
        #    )
        #self.layer_adap = adapblock(64, 448, shortcut=True, downsample=downsample)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        self._out_features = 512 * block.expansion

        self._init_params()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def _make_adap_kernel_layer(self, block, planes, blocks, stride=1, use_ibn=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_ibn=use_ibn))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, use_ibn=use_ibn))

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x, return_adapt_feat=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x1 = self.conv1(x)
        #x1 = self.bn1(x1)
        #x1 = self.relu(x1)
        #x2 = self.conv1_adap(x)
        #x2 = self.bn1_adap(x2)
        #x2 = self.relu(x2)
        #x =  x1 + x2
        x = self.maxpool(x)
        x = self.layer1(x)
        #x2 = self.layer1_adap(x)
        #x = x1 + x2 #torch.max(x1, x2)
        #x = self.layer_adap(x)
        x = self.layer2(x)
        #x2 = self.layer2_adap(x)
        #x = x1 + x2
        #x = self.layer_adap(x)
        x = self.layer3(x)
        #x2 = self.layer3_adap(x)
        #x = x1 + x2 #self.gate3(x1) + self.gate3(x2) #torch.max(x1, x2)
        #x = self.layer_adap(x)
        x1 = self.layer4(x)
        x2 = self.layer4_adap(x)
        x = x1 + x2 #self.gate4(x1) + self.gate4(x2) #torch.max(x1, x2)
        #x1 = self.layer4_new(x)
        #x2 = self.layer_adap(x1)
        #x = torch.cat([x1, x2], 1)
        if return_adapt_feat:
            return x, x2
        else:
            return x

    def forward(self, x):
        f = self.featuremaps(x)
        v = self.global_avgpool(f)
        return v.view(v.size(0), -1)


def init_pretrained_weights(model, model_url):
    pretrain_dict = model_zoo.load_url(model_url)
    model.load_state_dict(pretrain_dict, strict=False)


"""
Residual network configurations:
--
resnet18: block=BasicBlock, layers=[2, 2, 2, 2]
resnet34: block=BasicBlock, layers=[3, 4, 6, 3]
resnet50: block=Bottleneck, layers=[3, 4, 6, 3]
resnet101: block=Bottleneck, layers=[3, 4, 23, 3]
resnet152: block=Bottleneck, layers=[3, 8, 36, 3]
"""


@BACKBONE_REGISTRY.register()
def resnet18_adapkernel(pretrained=True, **kwargs):
    model = ResNet(block=BasicBlock, adapblock=AdapKernelBlock, layers=[2, 2, 2, 2])
    print('m={} for Adaptive Kernel without channel gate, no residual connection.'.format(reduction))
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model


@BACKBONE_REGISTRY.register()
def resnet34_adapkernel(pretrained=True, **kwargs):
    model = ResNet(block=BasicBlock, adapblock=AdapKernelBlock, layers=[3, 4, 6, 3])

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet34'])

    return model


@BACKBONE_REGISTRY.register()
def resnet50_adapkernel(pretrained=True, **kwargs):
    model = ResNet(block=Bottleneck, adapblock=AdapKernelBottleneck, layers=[3, 4, 6, 3])

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    return model


@BACKBONE_REGISTRY.register()
def resnet101_adapkernel(pretrained=True, **kwargs):
    model = ResNet(block=Bottleneck, adapblock=AdapKernelBottleneck, layers=[3, 4, 23, 3])
    print('m=16 for Adaptive Kernel without channel gate, no residual connection.')
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet101'])

    return model


@BACKBONE_REGISTRY.register()
def resnet152_adapkernel(pretrained=True, **kwargs):
    model = ResNet(block=Bottleneck, adapblock=AdapKernelBottleneck, layers=[3, 8, 36, 3])

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet152'])

    return model

