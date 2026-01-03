import cv2
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.model_zoo as model_zoo
from torch.distributions import Normal
import numpy as np

from .build import BACKBONE_REGISTRY
from .backbone import Backbone
from .mixstyle import MixStyle
from dassl.utils import mix_amplitude, copy_paste #, cutmix, rand_bbox


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def scale_cam_image(cam, target_size):
        result = []
        for img in cam:
            img = cv2.resize(img, target_size)
            result.append(img)
        result = np.float32(result)

        return result

class CutMix(nn.Module):
    def __init__(self, mix_beta=1., prob=0.5, cn_mix=False):
        super().__init__()
        self.mix_beta = mix_beta
        self.prob = prob
        self.cn_mix = cn_mix
        #self.mixstyle = MixStyle(128)#probablity=prob, alpha=mix_beta)

    def forward(self, x, label=None, domain_bs=None, cam=None):
        r = np.random.rand(1)
        if not self.training:
            return x

        width, height = x.size(-1), x.size(-2)
        cam = scale_cam_image(cam, target_size=(width, height))
        #x_mix, label1, lam, rand_index = cutmix(x, label, self.mix_beta)
        #x = channel_mix(x)
        #if r <= self.prob:
        #x_mix, label1, lam, rand_index = featmix(x, label, self.mix_beta)
        if r <= self.prob and self.cn_mix and False:
            x = channel_mix(x)#
        #x_mix, label1, lam, rand_index = feat_content_mix(x, self.mix_beta, label, cam=cam)
        x_mix, label1, lam, rand_index, mask = content_mix_cam(x, self.mix_beta, label, cam=cam)
        #else:
        #    rand_index = torch.arange(x.shape[0])
        #    if x.is_cuda:
        #        rand_index = rand_index.cuda()
        #    x_mix, label1, lam = x, label, 1.

        if r <= self.prob and self.cn_mix and False:
            #x_mix = channel_mix(x_mix)#, self.mix_beta)
            #x_disp = x_mix.clone().detach()
            #print(x)
            x_mix = channel_mix(x_mix, self.mix_beta)
            #print(x)
            #print(torch.sum(x_disp-x_mix.detach()), torch.sum(x_disp-x.detach()))
            
        return x_mix, label1, lam, rand_index


class FeatMix(nn.Module):
    def __init__(self, mix_beta=1., prob=1.):
        super().__init__()
        self.mix_beta = mix_beta
        self.prob = prob

    def forward(self, x, label=None, domain_bs=None):
        r = np.random.rand(1)
        if not self.training or r > self.prob:
            return x
        
        #r = np.random.rand(1)
        #if r > 0.5:
        #    # 50% probability do mix amplitude
        #    with torch.no_grad():
        #        rand_idx = torch.randperm(x.size(0)).cuda()
        #        x = mix_amplitude(x, x[rand_idx]).cuda()
        #    # if do mix amplitude, the label and index are the same as input
        #    label1 = label
        #    lam = 1.
        #    rand_index = torch.arange(x.size(0))
        #else:
        x, label1, lam, rand_index = feat_content_mix(x, self.mix_beta, label, domain_bs)
        return x, label1, lam, rand_index 
        

class FeatMixOut(nn.Module):
    def __init__(self, in_planes, mix_beta=.1, prob=1.):
        super().__init__()
        self.mix_beta = mix_beta
        self.prob = prob
        self.conv1 = conv3x3(in_planes, in_planes)
        self.conv2 = conv3x3(in_planes, in_planes)

    def forward(self, x, label=None, domain_bs=None):
        r = np.random.rand(1)
        if not self.training or r > self.prob:
            return x
        
        r = np.random.rand(1)
        lam = np.random.beta(self.mix_beta, self.mix_beta)
        #rand_idx = torch.randperm(x.size(0)).cuda()
        #x_mix = mix_amplitude(x, x[rand_idx]).cuda()
        x = self.conv1(x) * (1-lam) + self.conv2(x) * lam
        x = F.relu(x)
        label1 = label
        rand_index = torch.arange(x.size(0))
        return x, label1, lam, rand_index


def feat_content_mix(x, beta, label=None, domain_bs=None, cam=None):
        B, C, W, H = x.size()
        lam_region = np.random.beta(beta, beta)
        #lam_channel = np.random.beta(self.mix_beta, self.mix_beta)
        lam_channel = np.random.beta(1, 0.1)
        c_num = np.ceil(C * lam_channel + 1e-6).astype(np.int16)
        c_index = np.random.choice(C, c_num)
        #rand_index = np.random.permutation(np.arange(B))
        rand_index = torch.arange(B-1, -1, -1)
        # without this, there will be a error
        #one of the variables needed for gradient computation has been modified by an inplace operation
        x_mix = x.clone().detach() # detach x to avoid error
        x_mix = x_mix[rand_index, :, :, :]
        mask = torch.zeros_like(x)
        #with torch.no_grad():
        if label is not None:
            label1 = label[rand_index]
        else:
            label1 = None
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam_region)
        #x_tmp = x[rand_index, :, bbx1:bbx2, bby1:bby2]
        mask[:, :, bbx1:bbx2, bby1:bby2] = 1
        #import pdb
        #pdb.set_trace()
        if cam is not None:
            if isinstance(cam, np.ndarray):
                cam = torch.from_numpy(cam).unsqueeze(1)
            mask_content = cam[rand_index, :, :, :]
            mask_content = mask_content.bool()
            if x.is_cuda:
                mask_content = mask_content.cuda()
        else:
            mask_squeeze_channel = False
            if mask_squeeze_channel:
                mu = x_mix.mean(dim=[2, 3], keepdim=True)
                mask_content = (x_mix >= mu).float()
                mask_content = mask_content.sum(dim=1, keepdim=True)
                mu = mask_content.mean(dim=[2, 3], keepdim=True)
                mask_content = (mask_content >= mu)
            else:
                mu = x_mix.mean(dim=1, keepdim=True)
                mask_content = (x_mix >= mu)
                #mask_content = torch.clamp(mask_content.sum(dim=1, keepdim=True), 0., 1.)
        mask_content = mask_content.detach().float()
        # the mask should be randomly generated to include background, but the mixing ratio should be content oriented.
        #mask_mix_ratio = mask * mask_content
        mask *= mask_content
        mask_mix_ratio = mask
        #x_mix = x_mix[rand_index, :, :, :]
        x = x * (1-mask) + x_mix * mask
        # adjust lambda to exactly match pixel ratio
        #lam_region = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (H * W))
        #if mask_squeeze_channel:
        lam = 1 - mask_mix_ratio[:, 0, :, :].sum(dim=[-2, -1]) / (H * W)
        #else:
        #    lam = 1 - mask.sum(dim=[-2, -1]) / (C * H * W)
        
        #x_mix[:, c_index, bbx1:bbx2, bby1:bby2] = x_tmp[:, c_index, :, :]
        #lam = lam_region #* lam_channel
        #lam_mix = np.random.beta(self.mix_beta, self.mix_beta)
        #x = (1 - lam_mix) * x + lam_mix * x_mix
        #x_mix[:, c_index, bbx1:bbx2, bby1:bby2] = (1 - lam_mix) * x[:, c_index, bbx1:bbx2, bby1:bby2] + lam_mix * x_tmp[:, c_index, :, :]
        #lam = lam_region * lam_channel * lam_mix
        #print(lam, lam_region, lam_channel, lam_mix)
        return x, label1, lam, rand_index
        
def content_mix_cam(x, beta, label, cam, domain_bs=None):
    B, C, W, H = x.size()
    # without this, there will be a error
    #one of the variables needed for gradient computation has been modified by an inplace operation
    x_mix = x.clone().detach() # detach x to avoid error

    rand_index = np.random.permutation(np.arange(B))
    #if domain_bs is not None:
    #    rand_index = torch.arange(B-1, -1, -1)
    #    x_target = torch.split(x_mix, domain_bs)[-1]
    #    x_mix = x_target.repeat([len(domain_bs), 1, 1, 1])
    
    #x_mix_mean = x_mix.mean(dim=[2,3])
    #dist = compute_euclidean_dist(x_mix_mean)
    #rand_index = torch.argmax(dist, dim=0)

    x_mix = x_mix[rand_index, :, :, :]
    if isinstance(cam, np.ndarray):
        cam = scale_cam_image(cam, target_size=(W, H))
        # shape of mask_content: B*H*W, no channel, so increase the channel dim
        cam = torch.from_numpy(cam).unsqueeze(1)
    if x.is_cuda:
        cam = cam.cuda()
    mask_mix = cam[rand_index, :, :, :]
    mask_content, x = copy_paste(mask_mix, x_mix, cam, x)
    
    if label is not None:
        label1 = label[rand_index]
    else:
        label1 = None
    
    lam = 1 - mask_content[:, 0, :, :].sum(dim=[-2, -1]) / (H * W)

    return x, label1, lam, rand_index, mask_content
    
def compute_euclidean_dist(x):
    t1 = x.unsqueeze(1).expand(len(x), len(x), x.shape[1])
    t2 = x.unsqueeze(0).expand(len(x), len(x), x.shape[1])
    d = (t1 - t2).pow(2).sum(2)
    return d

def cutmix(input, target, beta, mask=None):
    '''
        image level CutMix
    '''
    lam = np.random.beta(beta, beta)
    rand_index = torch.randperm(input.size()[0])
    if input.is_cuda:
        rand_index = rand_index.cuda()
    target_a = target
    target_b = target[rand_index]
    # without this, there will be a error:
    # one of the variables needed for gradient computation has been modified by an inplace operation
    input_mix = input.clone()
    bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
    input_mix[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
    if mask is not None:
        mask = mask[rand_index]
        return input_mix, target_b, lam, mask
    return input_mix, target_b, lam, rand_index

def channel_mix(x, beta=0.1, rand_index=None):
    mu = x.mean(dim=[2, 3], keepdim=True) # compute instance mean
    mu = mu.detach()
    lam = np.random.beta(beta, beta)
    if rand_index is None:
        rand_index = torch.randperm(x.size()[0])
    mu_mix = mu[rand_index, :, :, :]
    mu_mix = lam * mu + (1. - lam) * mu_mix
    x = x * (mu_mix / mu)
    #mu_mix = (lam - 1.) * mu + (1. - lam) * mu_mix
    #x = x + mu_mix
    return x

def featmix(input, target, beta, mask_label=None):
    '''
        feature-level CutMix, or FeatureMix
    '''
    lam = np.random.beta(beta, beta)
    rand_index = torch.randperm(input.size()[0])
    if input.is_cuda:
        rand_index = rand_index.cuda()
    target_a = target
    target_b = target[rand_index]
    # without this, there will be a error:
    # one of the variables needed for gradient computation has been modified by an inplace operation
    input_mix = input.clone().detach()
    #mix_weight = np.random.beta(0.1, 0.1)
    mask = torch.zeros_like(input)
    bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
    #input_mix[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
    mask[:, :, bbx1:bbx2, bby1:bby2] = 1
    input_mix = input_mix[rand_index, :, :, :]
    #input = (1 - mix_weight) * input * (1 - mask) + input_mix * mask * mix_weight
    input = input * (1 - mask) + input_mix * mask  
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
    if mask_label is not None:
        mask_label = mask_label[rand_index]
        return input_mix, target_b, lam, mask_label
    #lam *= mix_weight
    return input, target_b, lam, rand_index


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


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


class ResNet(Backbone):

    def __init__(self, block, layers, **kwargs):
        self.inplanes = 64
        super().__init__()
        #self.fmix_cfg = (1, 0, 0, 0, 0)
        self.fmix_cfg = None
        self.fmix_beta  = 1.
        self.fmix_prob = 1.
        for k, v in kwargs.items():
            if 'fmix_cfg' in k:
                self.fmix_cfg = v
            if 'fmix_beta' in k:
                self.fmix_beta  = v
            if 'fmix_prob' in k:
                self.fmix_prob = v
        # backbone network
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        if True: #self.fmix_cfg[0] == 1:
            self.feat_mix0 = CutMix(mix_beta=self.fmix_beta,
                    prob=self.fmix_prob, cn_mix=False)
            #self.feat_mix0 = FeatMixOut(64, mix_beta=self.fmix_beta,
            #        prob=self.fmix_prob)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        if True: #self.fmix_cfg[1] == 1:
            self.feat_mix1 = CutMix(mix_beta=self.fmix_beta,
                                    prob=self.fmix_prob, cn_mix=True)
            #self.feat_mix1 = FeatMixOut(64, mix_beta=self.fmix_beta,
            #                         prob=self.fmix_prob)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        if True: #self.fmix_cfg[2] == 1:
            self.feat_mix2 = CutMix(mix_beta=self.fmix_beta,
                                     prob=self.fmix_prob, cn_mix=True)
            #self.feat_mix2 = FeatMixOut(128, mix_beta=self.fmix_beta,
            #                         prob=self.fmix_prob)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        if True: #self.fmix_cfg[3] == 1:
            self.feat_mix3 = CutMix(mix_beta=self.fmix_beta,
                                     prob=self.fmix_prob, cn_mix=True)
            #self.feat_mix3 = FeatMixOut(256, mix_beta=self.fmix_beta,
            #                        prob=self.fmix_prob)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        if True: #self.fmix_cfg[4] == 1:
            self.feat_mix4 = CutMix(mix_beta=self.fmix_beta,
                                     prob=self.fmix_prob, cn_mix=True)
            #self.feat_mix4 = FeatMixOut(512, mix_beta=self.fmix_beta,
            #                         prob=self.fmix_prob)

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        self._out_features = 512 * block.expansion

        self._init_params()
        #parameters for FeatMix
        print('config for FeatMix {}, beta {}'.format(self.fmix_cfg, self.fmix_beta))

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

    def featuremaps(self, x, label=None, domain_btcsize=None, use_fmix=True, cam=None):
        if self.fmix_cfg is None:
            idx = np.random.randint(0, 5)
            self.fmix_cfg = [0, 0, 0, 0, 0]
            self.fmix_cfg[idx] = 1
        assert sum(self.fmix_cfg)==1, 'only one element in TRAINER.FEATMIX.CONFIG can be 1'
        out_fmix = None
        #out_fmix1, out_fmix2 = None, None
        #out_fmix3, out_fmix4 = None, None
        if self.fmix_cfg[0] == 1 and use_fmix:
            out_fmix = self.feat_mix0(x, label=label, domain_bs=domain_btcsize, cam=cam)
            x = out_fmix[0]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        if self.fmix_cfg[1] == 1 and use_fmix and self.training:
            out_fmix = self.feat_mix1(x, label=label, domain_bs=domain_btcsize, cam=cam)
            x = out_fmix[0]
        x = self.layer2(x)
        if self.fmix_cfg[2] == 1 and use_fmix and self.training:
            out_fmix = self.feat_mix2(x, label=label, domain_bs=domain_btcsize, cam=cam)
            x = out_fmix[0]
        x = self.layer3(x)
        if self.fmix_cfg[3] == 1 and use_fmix and self.training:
            out_fmix = self.feat_mix3(x, label=label, domain_bs=domain_btcsize, cam=cam)
            x = out_fmix[0]
        x = self.layer4(x)
        if self.fmix_cfg[4] == 1 and use_fmix and self.training:
            out_fmix = self.feat_mix4(x, label=label, domain_bs=domain_btcsize, cam=cam)
            x = out_fmix[0]
        return x, out_fmix

    def forward(self, x, label=None, domain_btcsize=None, use_fmix=True, cam=None):
        f, fmix = self.featuremaps(
                x, 
                label=label,
                domain_btcsize=domain_btcsize, 
                use_fmix=use_fmix,
                cam=cam
                )
        v = self.global_avgpool(f)
        return v.view(v.size(0), -1), fmix


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
def resnet18_fmix_cam(pretrained=True, **kwargs):
    model = ResNet(block=BasicBlock, layers=[2, 2, 2, 2],
                   **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model


@BACKBONE_REGISTRY.register()
def resnet34_fmix_cam(pretrained=True, **kwargs):
    model = ResNet(block=BasicBlock, layers=[3, 4, 6, 3],
                   **kwargs)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet34'])

    return model


@BACKBONE_REGISTRY.register()
def resnet50_fmix_cam(pretrained=True, **kwargs):
    model = ResNet(block=Bottleneck, layers=[3, 4, 6, 3],
                   **kwargs)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    return model


@BACKBONE_REGISTRY.register()
def resnet101_fmix_cam(pretrained=True, **kwargs):
    model = ResNet(block=Bottleneck, layers=[3, 4, 23, 3],
                   **kwargs)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet101'])

    return model


@BACKBONE_REGISTRY.register()
def resnet152_fmix_cam(pretrained=True, **kwargs):
    model = ResNet(block=Bottleneck, layers=[3, 8, 36, 3],
                   **kwargs)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet152'])

    return model
