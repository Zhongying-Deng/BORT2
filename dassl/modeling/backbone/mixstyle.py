import torch
from torch.nn.modules.instancenorm import _InstanceNorm
from torch.distributions import Beta


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)
    

class MixStyle(_InstanceNorm):
  # x: input features of shape (B, C, H, W)
  # p: probabillity to apply MixStyle (default: 0.5)
  # alpha: hyper-parameter for the Beta distribution (default: 0.1)
  # eps: a small value added before square root for numerical stability (default: 1e-6)
    def __init__(
        self,
        num_features: int,
        num_domains: int = 4,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        probability: float = 0.5,
        alpha: float = 0.1,
      ) -> None:
        super(_InstanceNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.prob = probability
        self.alpha = alpha
        self.num_domains = num_domains
    
    def forward(self, x, domain_btcsize=None, use_domain_label=True, use_mix_style=True, lmda=None, prob=None):
        if not self.training or not use_mix_style:
            return x

        if torch.rand(1) > self.prob:
            return x

        B = x.size(0) # batch size

        mu = x.mean(dim=[2, 3], keepdim=True) # compute instance mean
        var = x.var(dim=[2, 3], keepdim=True) # compute instance variance
        sig = (var + self.eps).sqrt() # compute instance standard deviation
        mu, sig = mu.detach(), sig.detach() # block gradients
        x_normed = (x - mu) / sig # normalize input
        if lmda is None:
            lmda = Beta(self.alpha, self.alpha).sample((B, 1, 1, 1)) # sample instance-wise convex weights

        if use_domain_label and domain_btcsize is not None:
            # in this case, input x = [x?i, x?j]
            perm = torch.arange(B) # index
            # perm = torch.arange(B-1, -1, -1) # inverse index
            perm_i = torch.split(perm, domain_btcsize, 0)

            #tmp = []
            #ind_domain = torch.randperm(self.num_domains)
            #for ind, _ in enumerate(perm_i):
            #    index = ind_domain[ind]
            #    perm_j = perm_i[index][torch.randperm(B // 2)] # shuffling
            #    tmp.append(perm_j)
            #perm = torch.cat(tmp, 0) # concatenation
            tmp = []
            for ind, perm_j in enumerate(perm_i):
                #assert self.num_domains == len(domain_btcsize), \
                #        'the length of domain_btcsize {} should equal to num_domains {}'.format(
                #                len(domain_btcsize), self.num_domains)
                perm_j = perm_j[torch.randperm(domain_btcsize[ind])] # shuffling
                #print(perm_j.size(), B // self.num_domains, B, self.num_domains)
                tmp.append(perm_j)
            tmp.reverse()
            perm = torch.cat(tmp, 0) # concatenation
        else:
            perm = torch.randperm(B) # generate shuffling indices

        mu2, sig2 = mu[perm], sig[perm] # shuffling
        #if mu.is_cuda:
        #    lmda = lmda.cuda()
        lmda = lmda.to(mu.device)
        #print(mu.is_cuda, mu2.is_cuda, lmda.is_cuda)
        mu_mix = mu * lmda + mu2 * (1 - lmda) # generate mixed mean
        sig_mix = sig * lmda + sig2 * (1 - lmda) # generate mixed standard deviation
        return x_normed * sig_mix + mu_mix # denormalize input using the mixed statistics


class MixStyleMSDA(MixStyle):
    """
       MixStyle for Multi-source Unsupervised Domain Adaptation:
       - Use target statistics for source images, the target statistics keep unchanged.
       - Input
           x: input features, target samples should be at the end of x, 
              e.g. [src_1, ..., src_n, tar_1, ..., tar_n]
           domain_btcsize: batch size of each domain, can be used to split input x
                           len(domain_btcsize) == number_of_domains && sum(domain_btcsize) == x.size(0)
           use_domain_label: whether use domain labels or not
    """
    def forward(self, x, domain_btcsize=None, use_domain_label=True, use_mix_style=True, lmda=None, prob=None):
        if not self.training:
            return x

        if prob is not None:
            self.prob = prob

        if torch.rand(1) > self.prob:
            return x
            
        B = x.size(0) # batch size
        
        mu = x.mean(dim=[2, 3], keepdim=True) # compute instance mean
        var = x.var(dim=[2, 3], keepdim=True) # compute instance variance
        sig = (var + self.eps).sqrt() # compute instance standard deviation
        mu, sig = mu.detach(), sig.detach() # block gradients
        x_normed = (x - mu) / sig # normalize input
        if use_mix_style:
            #src_bs = sum(domain_btcsize[0:-1])
            if lmda is None:
                lmda = Beta(self.alpha, self.alpha).sample((B, 1, 1, 1)) # sample instance-wise convex weights
            if not isinstance(lmda, list) and not torch.is_tensor(lmda):
                lmda = torch.tensor([float(lmda) for _ in range(B)])
                lmda = lmda.view(B, 1, 1, 1)
            #lmda_tmp[0:src_bs] = lmda
            #lmda = lmda_tmp
            if use_domain_label and domain_btcsize is not None:
                #assert self.num_domains == len(domain_btcsize), 'the length of domain_btcsize should equal to num_domains'
                # in this case, input x = [x?i, x?j]
                perm = torch.arange(B) # index
                perm_i = torch.split(perm, domain_btcsize, 0)
            
                perm_tar = perm_i[-1]
                perm_src = perm_i[0]
                tmp = []
                lmda_new = []
                for ind, perm_j in enumerate(perm_i):
                    if ind < len(perm_i) -1:
                        size_ratio = domain_btcsize[ind] // domain_btcsize[-1] + 1
                        perm_t = torch.cat([perm_tar for _ in range(size_ratio)], 0)
                        perm_t = perm_t[:domain_btcsize[ind]]
                        lmda_ = torch.cat([lmda for _ in range(size_ratio)], 0)
                        lmda_ = lmda_[:domain_btcsize[ind]]
                        tgt_idx = torch.randperm(domain_btcsize[ind])
                        perm_j = perm_t[tgt_idx] # shuffling
                        lmda_new += lmda_[tgt_idx]
                    else:
                        #idx = randint(0, len(domain_btcsize) - 1)
                        #perm_src = perm_i[idx]
                        #perm_j = perm_src[torch.randperm(domain_btcsize[idx])]
                        perm_j = perm_tar
                        lmda_new += lmda
                    #print(perm_j.size(), B // self.num_domains, B, self.num_domains)
                    tmp.append(perm_j)
                #tmp.reverse()  # inverse index
                perm = torch.cat(tmp, 0) # concatenation
            else:
                perm = torch.randperm(B) # generate shuffling indices
                #print('No domain label {}'.format(B // self.num_domains))
            
            mu2, sig2 = mu[perm], sig[perm] # shuffling
            if mu.is_cuda:
                lmda = lmda.cuda()
            
            if domain_btcsize is not None:
                lmda = torch.cat(lmda_new, 0).unsqueeze(-1)
            #    tar_lmda = lmda[-domain_btcsize[-1]:, :, :, :]
            #    lmda[-domain_btcsize[-1]:, :, :, :] = 1. - tar_lmda
            
            mu_mix = mu * lmda + mu2 * (1 - lmda) # generate mixed mean
            sig_mix = sig * lmda + sig2 * (1 - lmda) # generate mixed standard deviation
        else:
            sig_mix = sig
            mu_mix = mu
        return x_normed * sig_mix + mu_mix # denormalize input using the mixed statistics


class MixStyleUDA(MixStyle):
    """
       Input x should contain equal number of source samples and target ones
       [src_1, ..., src_n, tar_1, ..., tar_n]
       Here, domain_btcsize is useless, just keep the consistency with MixStyle
    """
    def forward(self, x, domain_btcsize=None, use_domain_label=True, use_mix_style=True):
        if not self.training:
            return x

        if torch.rand(1) > self.prob:
            return x

        B = x.size(0) # batch size

        mu = x.mean(dim=[2, 3], keepdim=True) # compute instance mean
        var = x.var(dim=[2, 3], keepdim=True) # compute instance variance
        sig = (var + self.eps).sqrt() # compute instance standard deviation
        mu, sig = mu.detach(), sig.detach() # block gradients
        x_normed = (x - mu) / sig # normalize input
        if use_mix_style:
            lmda = Beta(self.alpha, self.alpha).sample((B, 1, 1, 1)) # sample instance-wise convex weights
            if use_domain_label and (B % 2 == 0):
                #assert B % 2 == 0, 'Source images should equal to target images in a mini-batch{}'.format(B % 2)
                #perm = torch.arange(B-1, -1, -1) # inverse index
                perm = torch.arange(B)
                perm_j, perm_i = perm.chunk(2) # separate indices
                #print(perm_j, perm_i)
                perm_j = perm_i[torch.randperm(B // 2)] # shuffling the target indexes for source data
                #perm_i = perm_i[torch.randperm(B // 2)] # shuffling for the target doamin data, as data augmentation for the target data
                perm = torch.cat([perm_j, perm_i], 0) # concatenation
            else:
                perm = torch.randperm(B) # generate shuffling indices
                #print('No domain label {}, {}'.format(B, x.shape))

            mu2, sig2 = mu[perm], sig[perm] # shuffling
            if mu.is_cuda:
                lmda = lmda.cuda()
            #print(mu.is_cuda, mu2.is_cuda, lmda.is_cuda)
            mu_mix = mu * lmda + mu2 * (1 - lmda) # generate mixed mean

            sig_mix = sig * lmda + sig2 * (1 - lmda) # generate mixed standard deviation
        else:
            sig_mix = sig
            mu_mix = mu
        return x_normed * sig_mix + mu_mix # denormalize input using the mixed statistics

    

