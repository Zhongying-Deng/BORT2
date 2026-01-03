"""
Modified from https://github.com/qq995431104/Copy-Paste-for-Semantic-Segmentation
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import pdb

def random_flip_horizontal(mask, img, p=0.5):
    inv_idx = torch.arange(img.size(3)-1, -1, -1).long()
    if np.random.random() < p:
        try:
            img = torch.fliplr(img)
            # shape of cam: B*C*H*W
            #mask = mask[:, :, ::-1]
            mask = torch.fliplr(mask)
        except:
            inv_idx = torch.arange(img.size(3)-1, -1, -1).long()
            img = img[:, :, :, inv_idx]
            mask = mask[:, :, :, inv_idx]
    return mask, img


def img_add(img_src, img_main, mask_src):
    _, c, h, w = img_main.shape
    # mask_src is a numpy array with shape of B*H*W
    #mask = torch.from_numpy(mask_src).unsqueeze(1) # increase a dim: channel
    #if img_main.is_cuda:
    #    mask = mask.cuda()
    
    sub_img01 = mask_src * img_src
    
    #mask_02 = cv2.resize(mask_src, (w, h), interpolation=cv2.INTER_NEAREST)
    #mask_02 = np.asarray(mask_02, dtype=np.uint8)
    #mask_02 = torch.from_numpy(mask_02).unsqueeze(1)
    #if img_main.is_cuda:
    #    mask_02 = mask_02.cuda()
    mask_02 = F.interpolate(mask_src, (h, w))
    sub_img02 = mask_02 * img_main
    img_src_masked = F.interpolate(sub_img01, (h, w))
    img_main = img_main - sub_img02 + img_src_masked
    return img_main


def rescale_src(mask_src, img_src, h, w):
    #b, h_src, w_src = mask_src.shape
    b, c, h_src, w_src = img_src.shape
    max_reshape_ratio = min(h / h_src, w / w_src)
    rescale_ratio = np.random.uniform(0.2, max_reshape_ratio)

    # reshape src img and mask
    rescale_h, rescale_w = int(h_src * rescale_ratio), int(w_src * rescale_ratio)
    #mask_src = cv2.resize(mask_src, (rescale_w, rescale_h),
    #                      interpolation=cv2.INTER_NEAREST)
    mask_src = F.interpolate(mask_src, (rescale_h, rescale_w))
    # mask_src = mask_src.resize((rescale_w, rescale_h), Image.NEAREST)
    img_src = F.interpolate(img_src, (rescale_h, rescale_w),
                         mode='bilinear')

    # set paste coord
    py = int(np.random.random() * (h - rescale_h))
    px = int(np.random.random() * (w - rescale_w))

    # paste src img and mask to a zeros background
    img_pad = torch.zeros((b, c, h_src, w_src))
    #mask_pad = np.zeros((b, h, w), dtype=np.uint8)
    mask_pad = torch.zeros((b, 1, h_src, w_src))
    img_pad[:, :, py:int(py + h_src * rescale_ratio), px:int(px + w_src * rescale_ratio)] = img_src
    mask_pad[:, :, py:int(py + h_src * rescale_ratio), px:int(px + w_src * rescale_ratio)] = mask_src

    return mask_pad, img_pad


def large_scale_jittering(mask, img, min_scale=0.8, max_scale=1.25):
    rescale_ratio = np.random.uniform(min_scale, max_scale)
    b, c, h, w = img.shape

    # rescale
    h_new, w_new = int(h * rescale_ratio), int(w * rescale_ratio)
    img = F.interpolate(img, (h_new, w_new), mode='bilinear')
    #mask = cv2.resize(mask, (w_new, h_new), interpolation=cv2.INTER_NEAREST)
    mask = F.interpolate(mask, (h_new, w_new))
    #mask = mask.bool().float()
    #print('rescale_ratio {}, mask after cv2.resize {}'.format(rescale_ratio, mask.shape))
    # mask = mask.resize((w_new, h_new), Image.NEAREST)

    # crop or padding
    x, y = int(np.random.uniform(0, abs(w_new - w))), int(np.random.uniform(0, abs(h_new - h)))
    if rescale_ratio <= 1.0:  # padding
        img_pad = torch.ones((b, c, h, w)) * 168 / 255
        #mask_pad = np.zeros((b, h, w), dtype=np.uint8)
        mask_pad = torch.zeros((b, 1, h, w))
        if img.is_cuda:
            img_pad = img_pad.cuda()
            mask_pad = mask_pad.cuda()
        #print('mask_pad {}'.format(mask_pad.shape))
        img_pad[:, :, y:y+h_new, x:x+w_new] = img
        mask_pad[:, :, y:y+h_new, x:x+w_new] = mask
        return mask_pad, img_pad
    else:  # crop
        img_crop = img[:, :, y:y+h, x:x+w]
        mask_crop = mask[:, :, y:y+h, x:x+w]
        #print('mask_crop {}'.format(mask_crop.shape))
        if img.is_cuda:
            img_crop = img_crop.cuda()
            mask_crop = mask_crop.cuda()
        return mask_crop, img_crop


def copy_paste(mask_src, img_src, mask_main, img_main):
    mask_src, img_src = random_flip_horizontal(mask_src, img_src)
    mask_main, img_main = random_flip_horizontal(mask_main, img_main)

    # LSJ£¬large_scale_jittering
    if True:
        mask_src, img_src = large_scale_jittering(mask_src, img_src, min_scale=0.1, max_scale=2.)
        mask_main, img_main = large_scale_jittering(mask_main, img_main)
    else:
        # rescale mask_src/img_src to less than mask_main/img_main's size
        _, c, h, w = img_main.shape
        mask_src, img_src = rescale_src(mask_src, img_src, h, w)
    #print(img_main.shape, img_src.shape)
    #print(mask_main.shape, mask_src.shape)
    img = img_add(img_src, img_main, mask_src)
    #mask = img_add(mask_src, mask_main, mask_src)

    return mask_src, img
