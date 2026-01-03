from .build import build_backbone, BACKBONE_REGISTRY # isort:skip
from .backbone import Backbone # isort:skip

from .vgg import vgg16
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .resnet_ca import resnet18_ca, resnet34_ca, resnet50_ca, resnet101_ca, resnet152_ca
from .resnet_ca_visualize import  resnet18_ca_visualize, resnet34_ca_visualize, resnet50_ca_visualize, resnet101_ca_visualize, resnet152_ca_visualize
from .resnet_ca_adapt_kernel import resnet18_ca_adapt_kernel, resnet34_ca_adapt_kernel, resnet50_ca_adapt_kernel, resnet101_ca_adapt_kernel, resnet152_ca_adapt_kernel
from .resnet_ca_fda import resnet18_ca_fda, resnet34_ca_fda, resnet50_ca_fda, resnet101_ca_fda, resnet152_ca_fda
from .resnet_ca_dist_net import resnet18_ca_dist_net, resnet34_ca_dist_net, resnet50_ca_dist_net, resnet101_ca_dist_net, resnet152_ca_dist_net
from .resnet_dida import resnet18_dida, resnet34_dida, resnet50_dida, resnet101_dida, resnet152_dida
from .resnet_adap_kernel import resnet18_adapkernel, resnet34_adapkernel, resnet50_adapkernel, resnet101_adapkernel, resnet152_adapkernel
from .resnet_adapt_kernel_mixstyle import resnet18_adaptkernel_ms, resnet34_adaptkernel_ms, resnet50_adaptkernel_ms, resnet101_adaptkernel_ms, resnet152_adaptkernel_ms
from .resnet_feat_mix import  resnet18_fmix, resnet34_fmix, resnet50_fmix, resnet101_fmix, resnet152_fmix
from .resnet_feat_mix_for_visualize import  resnet18_fmix_visualize, resnet34_fmix_visualize, resnet50_fmix_visualize, resnet101_fmix_visualize, resnet152_fmix_visualize
from .resnet_feat_mix_cam import  resnet18_fmix_cam, resnet34_fmix_cam, resnet50_fmix_cam, resnet101_fmix_cam, resnet152_fmix_cam
from .resnet_feat_mix_cam_bn import  resnet18_fmix_cam_bn, resnet34_fmix_cam_bn, resnet50_fmix_cam_bn, resnet101_fmix_cam_bn, resnet152_fmix_cam_bn
from .resnet_feat_mix_cam_bidirection import  resnet18_fmix_cam_bidirection, resnet34_fmix_cam_bidirection, resnet50_fmix_cam_bidirection, resnet101_fmix_cam_bidirection, resnet152_fmix_cam_bidirection
from .resnet_feat_mix_cam_dist_net import resnet18_fmix_cam_dist_net, resnet34_fmix_cam_dist_net, resnet50_fmix_cam_dist_net, resnet101_fmix_cam_dist_net, resnet152_fmix_cam_dist_net
from .resnet_dist_net import resnet18_dist_net, resnet34_dist_net, resnet50_dist_net, resnet101_dist_net, resnet152_dist_net
from .resnet_adap_kernel_dist_net import resnet18_adapkernel_dist_net, resnet34_adapkernel_dist_net, resnet50_adapkernel_dist_net, resnet101_adapkernel_dist_net, resnet152_adapkernel_dist_net

from .resnet_mixstyle import resnet18_ms, resnet34_ms, resnet50_ms, resnet101_ms, resnet152_ms
from .resnet_mixstyle_msda import resnet18_ms_msda, resnet34_ms_msda, resnet50_ms_msda, resnet101_ms_msda, resnet152_ms_msda
from .resnet_drt import resnet18_drt, resnet101_drt
from .resnet_drt_dist_net import resnet18_drt_dist_net, resnet101_drt_dist_net

from .resnet_mixstyle_dist_net import resnet18_ms_dist_net, resnet34_ms_dist_net, resnet50_ms_dist_net, resnet101_ms_dist_net, resnet152_ms_dist_net
from .resnet_nflow import resnet18_nflow, resnet34_nflow, resnet50_nflow, resnet101_nflow, resnet152_nflow
from .resnet_nflow_domain_class_mix import resnet18_nflow_dcmix, resnet34_nflow_dcmix, resnet50_nflow_dcmix, resnet101_nflow_dcmix, resnet152_nflow_dcmix
from .resnet_nflow_class_mix import resnet18_nflow_cmix, resnet34_nflow_cmix, resnet50_nflow_cmix, resnet101_nflow_cmix, resnet152_nflow_cmix
from .resnet_nflow_class_mix_ca import resnet18_nflow_cmix_ca, resnet34_nflow_cmix_ca, resnet50_nflow_cmix_ca, resnet101_nflow_cmix_ca, resnet152_nflow_cmix_ca
from .alexnet import alexnet
from .alexnet_raw import alexnet_raw
from .alexnet_ca import alexnet_ca
from .mobilenetv2 import mobilenetv2
from .wide_resnet import wide_resnet_28_2
from .wide_resnet_dida import wide_resnet_28_2_dida
from .cnn_digitsdg import cnn_digitsdg
from .efficientnet import (
    efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3,
    efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7
)
from .shufflenetv2 import (
    shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5,
    shufflenet_v2_x2_0
)
from .cnn_digitsingle import cnn_digitsingle
from .cnn_digitsingle_dida import cnn_digitsingle_adapt_kernel
from .cnn_digitsingle_dida_mixstyle import cnn_digitsingle_adapt_kernel_ms
from .cnn_digitsingle_mixstyle import cnn_digitsingle_ms
from .preact_resnet18 import preact_resnet18
from .cnn_digit5_m3sda import cnn_digit5_m3sda
from .cnn_digit5_m3sda_ca import cnn_digit5_m3sda_ca
from .cnn_digit5_m3sda_adapt_kernel import cnn_digit5_m3sda_adapt_kernel
from .cnn_digit5_m3sda_dida_mixstyle import cnn_digit5_m3sda_adapt_kernel_ms
from .cnn_digit5_m3sda_mixstyle import cnn_digit5_m3sda_ms
from .cnn_digit5_m3sda_mixstyle_dist_net import cnn_digit5_m3sda_ms_dist_net
from .cnn_digit5_m3sda_fmix_cam import cnn_digit5_m3sda_fmix_cam
from .cnn_digit5_m3sda_fmix_cam_dist_net import cnn_digit5_m3sda_fmix_cam_dist_net
from .cnn_digit5_m3sda_nflow_class_mix import cnn_digit5_m3sda_nflow_cmix
from .cnn_digit5_m3sda_nflow_class_mix_ca import cnn_digit5_m3sda_nflow_cmix_ca
from .wide_resnet_grc import wide_resnet_grc_28_2