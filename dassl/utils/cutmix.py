import torch
import numpy as np


def cutmix(input, target, beta, mask=None):
    lam = np.random.beta(beta, beta)
    rand_index = torch.randperm(input.size()[0])
    if input.is_cuda:
        rand_index = rand_index.cuda()
    target_a = target
    target_b = target[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
    input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
    if mask is not None:
        mask = mask[rand_index]
        return input, target_b, lam, mask
    return input, target_b, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_fixed_src_ratio(input_src, label_src, input_tgt, label_tgt, ratio, mask=None):
    #beta = 1.
    #lam = np.random.beta(beta, beta)
    #lam = min(lam, 1. - lam)
    lam = np.random.uniform(0., 0.1)
    #lam = np.random.beta(1, 5)
    bs_src = input_src.size(0)
    bs_tgt = input_tgt.size(0)
    if bs_tgt < bs_src:
        times = bs_src // bs_tgt
        input_tmp = torch.cat([input_tgt for _ in range(times + 1)], 0)
        input_tgt = input_tmp[:bs_src, :, :, :]
        label_tgt = torch.cat([label_tgt for _ in range(times + 1)], 0)
        label_tgt = label_tgt[:bs_src]
        if mask is not None:
            mask = torch.cat([mask for _ in range(times + 1)], 0)
            mask = mask[:bs_src]

    #rand_index = torch.randperm(bs_src)
    if torch.is_tensor(ratio):
       ratio = ratio.cpu().numpy()
    chosen_num = np.floor(bs_src * ratio + 1e-6).astype(np.int16)
    rand_index = np.random.choice(bs_src, chosen_num)
    target_b = label_src
    target_b[rand_index] = label_tgt[rand_index]
    if torch.is_tensor(lam):
        lam = lam.cpu().numpy()
    #print('lam {}'.format(lam))
    bbx1, bby1, bbx2, bby2 = rand_bbox_fixed_lam(input_src.size(), lam)
    lam_new = np.ones(bs_src)
    if bbx1 < bbx2 and bby1 < bby2:
        input_src[rand_index, :, bbx1:bbx2, bby1:bby2] = input_tgt[rand_index, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam = 1. - ((bbx2 - bbx1) * (bby2 - bby1) / (input_src.size()[-1] * input_src.size()[-2]))
        lam_new[rand_index] = 0.
    else:
        lam_new = 1.
    #print('adjusted lam {}, bbox {}, {}, {}, {}'.format(lam, bbx1, bby1, bbx2, bby2))
    lam_new = torch.from_numpy(lam_new).cuda()
    if mask is not None:
        mask_new = torch.ones(bs_src)
        if mask.is_cuda:
            mask_new = mask_new.cuda()
        mask_new[rand_index] = mask[rand_index]
        return input_src, target_b, lam_new, mask_new
    return input_src, target_b, lam_new, rand_index


def rand_bbox_fixed_lam(size, lam):
    W = size[2]
    H = size[3]
    center_w = W // 2
    center_h = H // 2
    cut_rat = np.sqrt(1. - lam)

    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    cut_center_w = cut_w // 2
    cut_center_h = cut_h // 2
    range_w = center_w - cut_center_w
    range_h = center_h - cut_center_h

    # uniform
    if range_w > 0:
        cx = np.random.randint(center_w - range_w, center_w + range_w)
    else:
        cx = center_w
    if range_h > 0:
        cy = np.random.randint(center_h - range_h, center_h + range_h)
    else:
        cy = center_h

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

