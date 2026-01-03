"""
Reference

https://github.com/VisionLearningGroup/VisionLearningGroup.github.io/tree/master/M3SDA
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from .build import BACKBONE_REGISTRY
from .backbone import Backbone
from dassl.utils import mix_amplitude, copy_paste


def scale_cam_image(cam, target_size):
        result = []
        for img in cam:
            img = cv2.resize(img, target_size)
            result.append(img)
        result = np.float32(result)

        return result

class ContentMix(nn.Module):
    def __init__(self, mix_beta=1., prob=0.5, cn_mix=False):
        super().__init__()
        self.mix_beta = mix_beta
        self.prob = prob
        self.cn_mix = cn_mix

    def forward(self, x, label=None, domain_bs=None, cam=None):
        r = np.random.rand(1)
        if not self.training:
            return x

        width, height = x.size(-1), x.size(-2)
        cam = scale_cam_image(cam, target_size=(width, height))
        x_mix, label1, lam, rand_index, mask = content_mix_cam(x, self.mix_beta, label, cam=cam)

        return x_mix, label1, lam, rand_index

        
def content_mix_cam(x, beta, label, cam, domain_bs=None):
    B, C, W, H = x.size()
    
    rand_index = np.random.permutation(np.arange(B))
    # without this, there will be a error
    #one of the variables needed for gradient computation has been modified by an inplace operation
    x_mix = x.clone().detach() # detach x to avoid error
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
    
    
class FeatureExtractor(Backbone):

    def __init__(self):
        super().__init__()
        self.feat_mix0 = ContentMix(mix_beta=1., prob=1., cn_mix=True)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(8192, 3072)
        self.bn1_fc = nn.BatchNorm1d(3072)
        self.fc2 = nn.Linear(3072, 2048)
        self.bn2_fc = nn.BatchNorm1d(2048)
        
        self.dist_net_conv = nn.Linear(3072, 2048)
        self.dist_net_bn = nn.BatchNorm1d(2048)
        
        self._out_features = 2048

    def _check_input(self, x):
        H, W = x.shape[2:]
        assert H == 32 and W == 32, \
            'Input to network must be 32x32, ' \
            'but got {}x{}'.format(H, W)

    def forward(self, x, label=None, domain_btcsize=None, use_fmix=True, cam=None):
        self._check_input(x)
        out_fmix = None

        if use_fmix:
            out_fmix = self.feat_mix0(x, label=label, domain_bs=domain_btcsize, cam=cam)
            x = out_fmix[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, stride=2, kernel_size=3, padding=1)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, stride=2, kernel_size=3, padding=1)
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), 8192)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        sig = F.relu(self.dist_net_bn(self.dist_net_conv(x)))
        sig += 1e-10
        sig = F.dropout(sig, training=self.training)
        x = F.relu(self.bn2_fc(self.fc2(x)))
        return x, out_fmix, sig.squeeze()


@BACKBONE_REGISTRY.register()
def cnn_digit5_m3sda_fmix_cam_dist_net(**kwargs):
    """
    This architecture was used for the Digit-5 dataset in:

        - Peng et al. Moment Matching for Multi-Source
        Domain Adaptation. ICCV 2019.
    """
    return FeatureExtractor()
