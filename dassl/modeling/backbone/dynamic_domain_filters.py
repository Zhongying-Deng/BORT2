import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class DynamicDomainFilters(nn.Module):
    def __init__(self, k, inplanes, planes, m=4, padding=None, stride=1):
        super(DynamicDomainFilters, self).__init__()
        self.k = k
        self.channel = inplanes
        self.conv = nn.Conv2d(self.channel, self.channel // m, 1, padding=0, bias=True)
        if padding is None:
            self.conv_adap_kernel = nn.Conv2d(self.channel // m, self.channel // m, self.k, padding=(self.k-1) // 2, stride=stride)
            #self.conv_adap_kernel = nn.Conv2d(self.channel // m, self.channel // m, self.k, padding=(self.k-1) // 2, stride=stride, groups=self.channel // m)
        else:
            self.conv_adap_kernel = nn.Conv2d(self.channel // m, self.channel // m, self.k, padding=padding, stride=stride)
            #self.conv_adap_kernel = nn.Conv2d(self.channel // m, self.channel // m, self.k, padding=padding, stride=stride, groups=self.channel // m)
        #self.conv_k = nn.Conv2d(1, self.channel // m, 3, padding=1, bias=True)
        self.conv_k = nn.Conv2d(1, self.channel // m, 1, padding=0, bias=True)
        #self.conv_for_g = nn.Conv2d(self.channel, self.channel // m, 1, padding=0, bias=True)
        self.fuse = nn.Conv2d(self.channel // m, planes, 1, padding=0, bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(k)
        self.max_pool = nn.AdaptiveMaxPool2d(k)
        #self.prelu = nn.PReLU()

    def forward(self, x, domain_btcsize=None):
        N, C, H, W = x.shape
        # [N * C/m * H * W]
        f = F.relu(self.conv(x))
        #if H > 7 and H <= 14:
        #    new_x = F.interpolate(x, scale_factor=0.5)
        #elif H > 14 and H <= 28:
        #    new_x = F.interpolate(x, scale_factor=0.25)
        #elif H > 28:
        #    new_x = F.interpolate(x, scale_factor=0.125)
        #else:
        #    new_x = x
        
        # [N * C/m * K * K]
        #g = self.prelu(self.conv(self.pool(x)))
        #g = F.relu(self.conv_for_g(self.pool(x)))
        g = F.relu(self.conv(self.max_pool(x))) #+ F.relu(self.conv(self.avg_pool(x)))
        #pooled_feat = torch.cat([self.maxpool(x), self.pool(x)], 1)
        #pooled_feat = torch.max(self.maxpool(x), self.pool(x))
        #g = F.relu(self.conv(pooled_feat))
        
        # [1 * C/m * H * W]
        f_list = torch.split(f, 1, 0)
        # f_list = torch.split(x, 1, 0)
        g_list = torch.split(g, 1, 0)

        out = []
        for i in range(N):
            # [1* C/m * H * W]
            f_one = f_list[i]
            # [C/m * 1 * K * K]
            g_one = g_list[i].squeeze(0).unsqueeze(1)
            # [C/m * C/m * K * K]
            g_k = self.conv_k(g_one)
            #g_k = F.relu(g_k)
            self.conv_adap_kernel.weight = nn.Parameter(g_k)

            # [1* C/m * H * W]
            o = self.conv_adap_kernel(f_one)
            # print('o shape {}'.format(o.shape))
            out.append(o)

        # [N * C/m * H * W]
        # print(len(out))
        y = torch.cat(out, dim=0)
        #y = F.relu(y)
        # print(y.shape)
        y = self.fuse(y)
        return y


class GlobalDynamicDomainFilters(nn.Module):
    def __init__(self, k, inplanes, planes, m=4, padding=None, stride=1):
        super(GlobalDynamicDomainFilters, self).__init__()
        self.k = k
        self.channel = inplanes
        self.conv = nn.Conv2d(self.channel, self.channel // m, 1, padding=0, bias=True)
        self.bn1 = nn.BatchNorm2d(self.channel // m)
        if padding is None:
            self.conv_adap_kernel = nn.Conv2d(self.channel // m, self.channel // m, self.k, padding=(self.k-1) // 2, stride=stride)
            #self.conv_adap_kernel = nn.Conv2d(self.channel // m, self.channel // m, self.k, padding=(self.k-1) // 2, stride=stride, groups=self.channel // m)
        else:
            self.conv_adap_kernel = nn.Conv2d(self.channel // m, self.channel // m, self.k, padding=padding, stride=stride)
            #self.conv_adap_kernel = nn.Conv2d(self.channel // m, self.channel // m, self.k, padding=padding, stride=stride, groups=self.channel // m)
        self.bn2 = nn.BatchNorm2d(self.channel // m)
        #self.conv_k = nn.Conv2d(1, self.channel // m, 3, padding=1, bias=True)
        self.conv_k = nn.Conv2d(1, self.channel // m, 1, padding=0, bias=True)
        #self.conv_for_g = nn.Conv2d(self.channel, self.channel // m, 1, padding=0, bias=True)
        self.conv_kernel = nn.Conv2d(1, k*k, 1, padding=0, bias=True)
        self.fuse = nn.Conv2d(self.channel // m, planes, 1, padding=0, bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        #self.prelu = nn.PReLU() 

    def forward(self, x, domain_btcsize=None):
        N, C, H, W = x.shape
        if domain_btcsize is not None:
            assert N == sum(domain_btcsize)
        else:
            # if no domain label is given, then take a whole batch as a domain
            domain_btcsize = [N]
        #print('domain batch size {}, x.shape {}'.format(domain_btcsize, x.shape))
        # [N * C/m * H * W]
        f = F.relu(self.conv(x))
        # [N * C/m * 1 * 1]
        g = F.relu(self.conv(self.avg_pool(x))) #+ F.relu(self.conv(self.max_pool(x)))
        #pdb.set_trace()
        # [N * 1 * C/m * 1]
        g_perm = g.permute(0, 2, 1, 3)
        # [N * k^2 * C/m * 1]
        kernel = self.conv_kernel(g_perm)
        #kernel = F.relu(kernel)
        # [N * 1 * C/m * k^2]
        kernel = kernel.permute(0, 3, 2, 1)
        #g = F.relu(self.conv(pooled_feat))

        # [1 * C/m * H * W]
        f_list = torch.split(f, 1, 0)
        # f_list = torch.split(x, 1, 0)
        g_list = torch.split(kernel, domain_btcsize, 0)
        #g_list = torch.split(kernel, 1, 0)
        domain_k_list = []
        #ind = 0
        for g in g_list:
            domain_k_list.append(torch.sum(g, 0, keepdim=True) / g.size(0))
            #print(domain_k_list[ind].shape)
            #ind += 1
        # domain_g_list with the length of N, 
        #   stores the generated domain filters for each sample
        domain_g_list = []
        for i, btcsize in enumerate(domain_btcsize):
            for j in range(btcsize):
                domain_g_list.append(domain_k_list[i])
        #print(len(domain_g_list), domain_g_list[0].shape, domain_g_list[domain_btcsize[0]].shape, domain_g_list[domain_btcsize[0]+1].shape)
        #print(domain_g_list[0]-domain_g_list[domain_btcsize[0]])
        #print(domain_g_list[domain_btcsize[0]]-domain_g_list[domain_btcsize[0]+1])
        out = []
        a = True
        for i in range(N):
            # [1* C/m * H * W]
            f_one = f_list[i]
            # [1 * 1 * C/m * k^2]
            #g_one = g_list[i]
            g_one = domain_g_list[i]
            # Also consider following form [C/m * 1 * 1 * k^2]
            #g_one = g_list[i].permute(2, 0, 1, 3)
            if a:
                # [1 * C/m * C/m * k^2], also consider use 3*3 conv_k 
                #  to utilize relationship among each channel and kernel
                g_k = self.conv_k(g_one)
                g_k = F.relu(g_k)
                # [C/m * C/m * k * k]
                g_k = g_k.reshape(g_k.size(1), g_k.size(2), self.k, self.k)
            else:
                # [C/m * 1 * 1 * k^2]
                g_one = g_one.permute(2, 0, 1, 3)
                # [C/m * 1 * k * k]
                g_one = g_one.reshape(g_one.size(0), g_one.size(1), self.k, self.k)
                # [C/m * C/m * k * k]
                g_k = self.conv_k(g_one)
                g_k = F.relu(g_k).permute(1, 0, 2, 3)

            self.conv_adap_kernel.weight = nn.Parameter(g_k)

            # [1* C/m * H * W]
            o = self.conv_adap_kernel(f_one)
            # print('o shape {}'.format(o.shape))
            out.append(o)

        # [N * C/m * H * W]
        y = torch.cat(out, dim=0)
        #y = F.relu(y)
        y = self.fuse(y)
        return y

class MultiscaleGlobalDynamicDomainFilters(nn.Module):
    def __init__(self, k, inplanes, planes, m=4, padding=None, stride=1):
        super(MultiscaleGlobalDynamicDomainFilters, self).__init__()
        self.k = k
        self.channel = inplanes
        self.conv = nn.Conv2d(self.channel, self.channel // m, 1, padding=0, bias=True)
        self.bn1 = nn.BatchNorm2d(self.channel // m)
        if padding is None:
            #self.conv_adap_kernel = nn.Conv2d(self.channel // m, self.channel // m, self.k, padding=(self.k-1) // 2, stride=stride)
            self.conv_adap_kernel = nn.Conv2d(self.channel // m, self.channel // m, self.k, padding=(self.k-1) // 2, stride=stride, groups=self.channel // m)
            self.conv3x3_atrous_adap_kernel = nn.Conv2d(self.channel // m, self.channel // m, 3, padding=2, stride=stride, dilation=2, groups=self.channel // m)
        else:
            #self.conv_adap_kernel = nn.Conv2d(self.channel // m, self.channel // m, self.k, padding=padding, stride=stride)
            self.conv_adap_kernel = nn.Conv2d(self.channel // m, self.channel // m, self.k, padding=padding, stride=stride, groups=self.channel // m)
            self.conv3x3_atrous_adap_kernel = nn.Conv2d(self.channel // m, self.channel // m, 3, padding=2, stride=stride, dilation=2, groups=self.channel // m)
        self.conv1x1_adap_kernel = nn.Conv2d(self.channel // m, self.channel // m, 1, groups=self.channel // m)
        #self.conv_k = nn.Conv2d(1, self.channel // m, 3, padding=1, bias=True)
        self.conv_k = nn.Conv2d(2, 1, 1, padding=0, bias=True)
        self.conv_k2 = nn.Conv2d(2, 1, 1, padding=0, bias=True)
        #self.conv_for_g = nn.Conv2d(self.channel, self.channel // m, 1, padding=0, bias=True)
        self.conv_kernel = nn.Conv2d(1, k*k, 1, padding=0, bias=True)
        self.conv_kernel2 = nn.Conv2d(1, 3*3, 1, padding=0, bias=True)
        self.fuse = nn.Conv2d(self.channel * 2 // m, planes, 1, padding=0, bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        #self.prelu = nn.PReLU()

    def forward(self, x, domain_btcsize=None):
        N, C, H, W = x.shape
        if domain_btcsize is not None:
            assert N == sum(domain_btcsize)
        else:
            # if no domain label is given, then take a whole batch as a domain
            domain_btcsize = [N]
        #print('domain batch size {}, x.shape {}'.format(domain_btcsize, x.shape))
        # [N * C/m * H * W]
        f = F.relu(self.conv(x))
        # [N * C/m * 1 * 1]
        g = F.relu(self.conv(self.avg_pool(x))) #+ F.relu(self.conv(self.max_pool(x)))
        #pdb.set_trace()
        # [N * 1 * C/m * 1]
        g_perm = g.permute(0, 2, 1, 3)
        # [N * k^2 * C/m * 1]
        kernel = self.conv_kernel(g_perm)
        #kernel = F.relu(kernel)
        # [N * 1 * C/m * k^2]
        kernel = kernel.permute(0, 3, 2, 1)
        # [N * 3^2 * C/m * 1]
        kernel_atrous = self.conv_kernel2(g_perm)
        # [N * 1 * C/m * 3^2]
        kernel_atrous = kernel_atrous.permute(0, 3, 2, 1)

        # [1 * C/m * H * W]
        f_list = torch.split(f, 1, 0)

        g_list = torch.split(kernel, domain_btcsize, 0)
        g_one_list = torch.split(kernel, 1, 0)
        g_list_atrous = torch.split(kernel_atrous, domain_btcsize, 0)
        g_one_list_atrous = torch.split(kernel_atrous, 1, 0)

        domain_k_list = []
        domain_k_list_atrous = []
        for g in g_list:
            domain_k_list.append(torch.sum(g, 0, keepdim=True) / g.size(0))
        for g in g_list_atrous:
            domain_k_list_atrous.append(torch.sum(g, 0, keepdim=True) / g.size(0))
        # domain_g_list with the length of N,
        #   stores the generated domain filters for each sample
        domain_g_list = []
        domain_g_list_atrous = []
        for i, btcsize in enumerate(domain_btcsize):
            for j in range(btcsize):
                domain_g_list.append(domain_k_list[i])
                domain_g_list_atrous.append(domain_k_list_atrous[i])
        out = []
        out_atrous = []
        a = True
        for i in range(N):
            # [1* C/m * H * W]
            f_one = f_list[i]
            # [1 * 1 * C/m * k^2]
            g_img_one = g_one_list[i]
            g_domain_one = domain_g_list[i]
            # [1 * 2 * C/m * k^2]
            g_one =  torch.cat([g_img_one,g_domain_one], 1) # 0.5 * (g_img_one + g_domain_one)
            # Also consider following form [C/m * 1 * 1 * k^2]
            #g_one = g_list[i].permute(2, 0, 1, 3)
            g_img_one_atrous = g_one_list_atrous[i]
            g_domain_one_atrous = domain_g_list_atrous[i]
            g_one_atrous =  torch.cat([g_img_one_atrous, g_domain_one_atrous], 1)
            if a:
                # [1 * 1 * C/m * k^2], also consider use 3*3 conv_k
                #  to utilize relationship among each channel and kernel
                g_k = self.conv_k(g_one)
                g_k = F.relu(g_k)
                # [C/m * 1 * k * k]
                g_k = g_k.reshape(g_k.size(2), g_k.size(1), self.k, self.k)
                # [1 * 1 * C/m * 3^2]
                g_k_atrous = self.conv_k2(g_one_atrous)
                g_k_atrous = F.relu(g_k_atrous)
                # [C/m * 1 * 3 * 3]
                g_k_atrous = g_k_atrous.reshape(g_k_atrous.size(2), g_k_atrous.size(1), 3, 3)
            else:
                # [C/m * 2 * 1 * k^2]
                g_one = g_one.permute(2, 1, 0, 3)
                # [C/m * 2 * k * k]
                g_one = g_one.reshape(g_one.size(0), g_one.size(1), self.k, self.k)
                # [C/m * 1 * k * k]
                g_k = self.conv_k(g_one)
                g_k = F.relu(g_k)
                # [C/m * 2 * 1 * 3^2]
                g_one_atrous = g_one_atrous.permute(2, 1, 0, 3)
                # [C/m * 2 * 3 * 3]
                g_one_atrous = g_one_atrous.reshape(g_one_atrous.size(0), g_one_atrous.size(1), 3, 3)
                # [C/m * 1 * 3 * 3]
                g_k_atrous = self.conv_k2(g_one_atros)
                g_k_atrous = F.relu(g_k_atrous)

            self.conv_adap_kernel.weight = nn.Parameter(g_k)
            self.conv3x3_atrous_adap_kernel.weight = nn.Parameter(g_k_atrous)

            # [1* C/m * H * W]
            o = self.conv_adap_kernel(f_one)
            # print('o shape {}'.format(o.shape))
            out.append(o)
            o_atrous = self.conv3x3_atrous_adap_kernel(f_one)
            out_atrous.append(o_atrous)

        # [N * C/m * H * W]
        y = torch.cat(out, dim=0)
        #y = F.relu(y)
        y_atrous = torch.cat(out_atrous, dim=0)
        # [N * 2C/m * H * W]
        y = torch.cat([y, y_atrous], dim=1)
        y = self.fuse(y)
        return y


