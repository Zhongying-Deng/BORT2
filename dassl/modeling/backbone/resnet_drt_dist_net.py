import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torch.autograd import Function
import torch.nn.functional as F
import pdb

from .build import BACKBONE_REGISTRY
from .backbone import Backbone


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 3.

class SEModule(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class GradReverse(Function):
    #@staticmethod
    #def __init__(self, lambd):
    #    self.lambd = lambd
    @staticmethod
    def forward(grev, x, lmbd):
        #self.lmbd = lmbd
        grev.lmbd = lmbd
        return x.view_as(x)

    @staticmethod
    def backward(grev, grad_output):
        #print (grev.lmbd)
        return (grad_output*-grev.lmbd), None

def grad_reverse(x,lambd=1.0):
    return GradReverse.apply(x,lambd)

class ResNetDRA(Backbone):

    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, pretrain=None):
        super(ResNetDRA, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.dist_net_conv = nn.Conv2d(
                512 * block.expansion, 512 * block.expansion, 
                kernel_size=7, stride=1, padding=0, bias=True)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._out_features = 512 * block.expansion
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                #nn.init.constant_(m.weight, 1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
   
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        sig = self.dist_net_conv(x)
        sig += 1e-10
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x, sig.squeeze()

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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


class BasicBlockKDRA(nn.Module):
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

        inp = inplanes
        self.squeeze = inp // 16
        self.dim = int(math.sqrt(inp))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(inp, self.squeeze, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.squeeze, 4, bias=False),
        )
        self.sf = nn.Softmax(dim=1)
        self.conv_s1 = nn.Conv2d(planes, planes, 1, stride=1, padding=0, bias=False)
        self.conv_s2 = nn.Conv2d(planes, planes, 1, stride=1, padding=0, bias=False)
        self.conv_s3 = nn.Conv2d(planes, planes, 1, stride=1, padding=0, bias=False)
        self.conv_s4 = nn.Conv2d(planes, planes, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        residual = x

        b, c, h, w = x.size()
        y = self.fc(self.avg_pool(x).view(b, c)).view(b, 4, 1, 1, 1)
        y = self.sf(y)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        dyres = self.conv_s1(out)*y[:,0] + self.conv_s2(out)*y[:,1] + \
            self.conv_s3(out)*y[:,2] + self.conv_s4(out)*y[:,3]
        #print(out.shape, dyres.shape, self.conv2(out).shape)
        out = dyres + self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BottleneckKDRA(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckKDRA, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        inp, oup = inplanes, planes * 4
        self.squeeze = inp // 16
        self.dim = int(math.sqrt(inp))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(inp, self.squeeze, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.squeeze, 4, bias=False),
        )
        self.sf = nn.Softmax(dim=1)
        self.conv_s1 = nn.Conv2d(planes, planes, 1, stride=stride, padding=0, bias=False)
        self.conv_s2 = nn.Conv2d(planes, planes, 1, stride=stride, padding=0, bias=False)
        self.conv_s3 = nn.Conv2d(planes, planes, 1, stride=stride, padding=0, bias=False)
        self.conv_s4 = nn.Conv2d(planes, planes, 1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.fc(self.avg_pool(x).view(b, c)).view(b, 4, 1, 1, 1)
        y = self.sf(y)

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        dyres = self.conv_s1(out)*y[:,0] + self.conv_s2(out)*y[:,1] + \
            self.conv_s3(out)*y[:,2] + self.conv_s4(out)*y[:,3]
        out = dyres + self.conv2(out)

        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual #+ dyres
        out = self.relu(out)

        return out


class Classifier(nn.Module):
    def __init__(self, num_classes=13,num_unit=2048):
        super(Classifier, self).__init__()
        self.classifier = nn.Linear(num_unit, num_classes)

    def forward(self, x,reverse=False):
        x = self.classifier(x)
        return x


class ResClassifier(nn.Module):
    def __init__(self, num_classes=13,num_layer = 2,num_unit=2048,prob=0.5,middle=1000):
        super(ResClassifier, self).__init__()
        layers = []
        layers.append(nn.Dropout(p=prob))
        layers.append(nn.Linear(num_unit,middle))
        layers.append(nn.BatchNorm1d(middle,affine=True))
        layers.append(nn.ReLU(inplace=True))

        for i in range(num_layer-1):
            layers.append(nn.Dropout(p=prob))
            layers.append(nn.Linear(middle,middle))
            layers.append(nn.BatchNorm1d(middle,affine=True))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(middle,num_classes))
        self.classifier = nn.Sequential(*layers)

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x,reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        x = self.classifier(x)
        return x

def get_params(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    init_params = model.init_param
    for name, param in model.named_parameters():
        if param.requires_grad:
            if name in init_params:
                yield param


def get_10xparams(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    new_params = model.new_param
    for name, param in model.named_parameters():
        if param.requires_grad:
            if name in new_params:
                yield param


def init_pretrained_weights(model, model_url):
    pretrain_dict = model_zoo.load_url(model_url)
    model.load_state_dict(pretrain_dict, strict=False)


@BACKBONE_REGISTRY.register()
def resnet18_drt_dist_net(pretrained=True, **kwargs):
    model = ResNetDRA(BasicBlockKDRA, [2, 2, 2, 2])
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model


@BACKBONE_REGISTRY.register()
def resnet101_drt_dist_net(pretrained=True, **kwargs):
    model = ResNetDRA(BottleneckKDRA, [3, 4, 23, 3])
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet101'])

    return model

