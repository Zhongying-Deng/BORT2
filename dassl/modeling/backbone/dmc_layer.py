import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class GlobalAdaptiveKernel(nn.Module):
    def __init__(self, in_planes, out_planes, ratio=16, activation='relu'):
        super(GlobalAdaptiveKernel, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, out_planes, 1, bias=False)
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU(num_parameters=out_planes)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'linear':
            self.activation = None

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        if self.activation is not None:
            out = self.activation(out)
        return out


class DCMLayer(nn.Module):
    def __init__(self, k, inplanes, planes, m=4, padding=None, stride=1):
        super(DCMLayer, self).__init__()
        self.k = k
        self.channel = inplanes
        self.conv = nn.Conv2d(self.channel, self.channel // m, 1, padding=0, bias=True)
        if padding is None:
            self.dw_conv = nn.Conv2d(self.channel // m, self.channel // m, self.k, padding=(self.k-1) // 2, stride=stride, groups=self.channel // m)
        else:
            self.dw_conv = nn.Conv2d(self.channel // m, self.channel // m, self.k, padding=padding, stride=stride, groups=self.channel // m)

        self.fuse = nn.Conv2d(self.channel // m, planes, 1, padding=0, bias=True)
        self.pool = nn.AdaptiveAvgPool2d(k)

    def forward(self, x):
        N, C, H, W = x.shape
        # [N * C/m * H * W]
        f = F.relu(self.conv(x))
        # [N * C/m * K * K]
        #g = F.tanh(self.conv(self.pool(x)))
        g = self.conv(self.pool(x))

        f_list = torch.split(f, 1, 0)
        # f_list = torch.split(x, 1, 0)
        g_list = torch.split(g, 1, 0)

        out = []
        for i in range(N):
            # [1* C/m * H * W]
            f_one = f_list[i]
            # [C/m * 1 * K * K] for depthwise conv, same shape as self.dw_conv.weight
            g_one = g_list[i].squeeze(0).unsqueeze(1)
            #print(g_one.shape, self.dw_conv.weight.shape)
            self.dw_conv.weight = nn.Parameter(g_one)

            # [1* C/m * H * W]
            o = self.dw_conv(f_one)
            # print('o shape {}'.format(o.shape))
            out.append(o)

        # [N * C/m * H * W]
        # print(len(out))
        y = torch.cat(out, dim=0)
        # print(y.shape)
        y = self.fuse(y)
        return y


class AdaptiveKernelModule(nn.Module):
    def __init__(self, k, inplanes, planes, m=4, padding=None, stride=1):
        super(AdaptiveKernelModule, self).__init__()
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

    def forward(self, x):
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


class AdaptiveKernelFC(nn.Module):
    def __init__(self, k, inplanes, planes, m=4, padding=None, stride=1):
        super(AdaptiveKernelFC, self).__init__()
        self.k = k
        self.channel = inplanes
        #self.conv = nn.Conv2d(self.channel, self.channel // m, 1, padding=0, bias=True)
        if padding is None:
            self.conv_adap_kernel = nn.Conv2d(inplanes, planes, self.k, padding=(self.k-1) // 2, stride=stride)
        else:
            self.conv_adap_kernel = nn.Conv2d(inplanes, planes, self.k, padding=padding, stride=stride)
        #self.conv_k = nn.Conv2d(1, self.channel // m, 3, padding=1, bias=True)
        self.conv_k = nn.Conv2d(1, planes, 1, padding=0, bias=True)
        #self.conv_for_g = nn.Conv2d(self.channel, self.channel // m, 1, padding=0, bias=True)
        #self.fuse = nn.Conv2d(self.channel // m, planes, 1, padding=0, bias=True)
        #self.pool = nn.AdaptiveAvgPool2d(k)
        #self.pool = nn.AdaptiveMaxPool2d(k)
        #self.prelu = nn.PReLU()

    def forward(self, x):
        N, C, H, W = x.shape
        # [N * C/m * H * W]

        # [1 * C/m * H * W]
        f_list = torch.split(x, 1, 0)
        # f_list = torch.split(x, 1, 0)
        g_list = torch.split(x, 1, 0)

        out = []
        for i in range(N):
            # [1* C/m * H * W]
            f_one = f_list[i]
            # [C/m * 1 * K * K]
            g_one = g_list[i].squeeze(0).unsqueeze(1)
            # [C/m * C/m * K * K]
            g_k = self.conv_k(g_one)
            #print(g_one.shape, self.dw_conv.weight.shape)
            self.conv_adap_kernel.weight = nn.Parameter(g_k.permute(1, 0, 2, 3))

            # [1* C/m * H * W]
            o = self.conv_adap_kernel(f_one)
            # print('o shape {}'.format(o.shape))
            out.append(o)

        # [N * C/m * H * W]
        # print(len(out))
        y = torch.cat(out, dim=0)
        #y = F.relu(y)
        # print(y.shape)
        #y = self.fuse(y)
        return y


class GlobalAdaptKernelModule(nn.Module):
    def __init__(self, k, inplanes, planes, m=4, padding=None, stride=1):
        super(GlobalAdaptKernelModule, self).__init__()
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

    def forward(self, x):
        N, C, H, W = x.shape
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
        g_list = torch.split(kernel, 1, 0)

        out = []
        a = True 
        for i in range(N):
            # [1* C/m * H * W]
            f_one = f_list[i]
            # [1 * 1 * C/m * k^2]
            g_one = g_list[i]
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


class GlobalSpatialAdaptKernelModule(nn.Module):
    def __init__(self, k, inplanes, planes, m=4, padding=None, stride=1):
        super(GlobalSpatialAdaptKernelModule, self).__init__()
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
        self.generate_kernel_tyle = True
        if self.generate_kernel_tyle:
            self.conv_k = nn.Conv2d(2, self.channel // m, 1, padding=0, bias=True)
        else:
            self.conv_k = nn.Conv2d(1, self.channel // (2*m), 1, padding=0, bias=True)
            self.conv_k2 = nn.Conv2d(1, self.channel // (2*m), 1, padding=0, bias=True)
        #self.conv_k = nn.Conv2d(1, self.channel // m, 3, padding=1, bias=True)
        #self.conv_for_g = nn.Conv2d(self.channel, self.channel // m, 1, padding=0, bias=True)
        self.conv_kernel = nn.Conv2d(1, k*k, 1, padding=0, bias=True)
        self.fuse = nn.Conv2d(self.channel // m, planes, 1, padding=0, bias=True)
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.max_pool1 = nn.AdaptiveMaxPool2d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(k)
        self.max_pool2 = nn.AdaptiveMaxPool2d(k)

    def forward(self, x):
        N, C, H, W = x.shape
        # [N * C/m * H * W]
        f = F.relu(self.conv(x))
        # [N * C/m * 1 * 1], generate kernel based on channel-based global information
        global_k = F.relu(self.conv(self.avg_pool1(x))) #+ F.relu(self.conv(self.max_pool(x)))
        #pdb.set_trace()
        # [N * 1 * C/m * 1]
        g_perm = global_k.permute(0, 2, 1, 3)
        # [N * k^2 * C/m * 1]
        global_kernel = self.conv_kernel(g_perm)
        #kernel = F.relu(kernel)
        # [N * 1 * C/m * k^2]
        global_kernel = global_kernel.permute(0, 3, 2, 1)
        #g = F.relu(self.conv(pooled_feat))
        
        # [N * C/m * K * K], generate kernel based on spatial information
        spatial_k = F.relu(self.conv(self.max_pool2(x)))
        # [1 * C/m * H * W]
        f_list = torch.split(f, 1, 0)
        # f_list = torch.split(x, 1, 0)
        global_list = torch.split(global_kernel, 1, 0)
        spatial_list = torch.split(spatial_k, 1, 0)

        out = []
        for i in range(N):
            # [1* C/m * H * W]
            f_one = f_list[i]
            # [1 * 1 * C/m * k^2]
            g_one = global_list[i]
            # [1 * C/m * K * K] permute to [C/m * 1 * K * K]
            s_one = spatial_list[i].permute(1, 0, 2, 3)
            
            if self.generate_kernel_tyle:
                # [C/m * 1 * 1 * k^2]
                g_one = g_one.permute(2, 0, 1, 3)
                # [C/m * 1 * k * k]
                g_one = g_one.reshape(g_one.size(0), g_one.size(1), self.k, self.k)
                # [C/m * 2 * k * k] concatenate channel and spatial infomation
                gs_one = torch.cat([g_one, s_one], dim=1)
                # [C/m * C/m * k * k]
                g_k = self.conv_k(gs_one)
                g_k = F.relu(g_k)
                g_k = g_k.permute(1, 0, 2, 3)

            else:
                # [1 * C/2m * C/m * k^2]
                glbl_k = self.conv_k(g_one)
                # [1 * C/m * C/2m * k^2]
                glbl_k = glbl_k.permute(0, 2, 1, 3)
                # [C/m * C/2m * K * K]
                glbl_k = glbl_k.reshape(glbl_k.size(1), glbl_k.size(2), self.k, self.k)
                glbl_k = F.relu(glbl_k)
                # [C/m * C/2m * K * K]
                sptl_k = self.conv_k2(s_one)
                # or sptl_k share the kernel with glbl_k: sptl_k = self.conv_k(s_one)
                # [C/m * C/m * K * K]
                g_k = torch.cat([glbl_k, sptl_k], dim=1)

                #g_k = F.relu(g_k)
                #g_k = g_k.permute(1, 0, 2, 3)

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


class GlobalMultiScaleAdaptKernelModule(nn.Module):
    def __init__(self, k, inplanes, planes, m=4, padding=None, stride=1):
        super(GlobalMultiScaleAdaptKernelModule, self).__init__()
        self.k = k
        self.channel = inplanes
        self.conv = nn.Conv2d(self.channel, self.channel // m, 1, padding=0, bias=True)
        #self.bn1 = nn.BatchNorm2d(self.channel // m)
        if padding is None:
            #self.conv_adap_kernel = nn.Conv2d(self.channel // m, self.channel // m, self.k, padding=(self.k-1) // 2, stride=stride)
            self.conv_adap_kernel = nn.Conv2d(self.channel // m, self.channel // m, self.k, padding=(self.k-1) // 2, stride=stride, groups=self.channel // m)
            self.conv3x3_atrous_adap_kernel = nn.Conv2d(self.channel // m, self.channel // m, 3, padding=2, stride=stride, dilation=2, groups=self.channel // m)
        else:
            #self.conv_adap_kernel = nn.Conv2d(self.channel // m, self.channel // m, self.k, padding=padding, stride=stride)
            self.conv_adap_kernel = nn.Conv2d(self.channel // m, self.channel // m, self.k, padding=padding, stride=stride, groups=self.channel // m)
            self.conv3x3_atrous_adap_kernel = nn.Conv2d(self.channel // m, self.channel // m, 3, padding=2, stride=stride, dilation=2, groups=self.channel // m)
        #self.bn2 = nn.BatchNorm2d(self.channel // m)
        #self.conv_k = nn.Conv2d(1, self.channel // m, 3, padding=1, bias=True)
        self.conv_k = nn.Conv2d(1, 1, 1, padding=0, bias=True)
        self.conv_k2 = nn.Conv2d(1, 1, 1, padding=0, bias=True)
        #self.conv_for_g = nn.Conv2d(self.channel, self.channel // m, 1, padding=0, bias=True)
        self.conv_kernel = nn.Conv2d(1, k*k, 1, padding=0, bias=True)
        self.conv_kernel2 = nn.Conv2d(1, 3*3, 1, padding=0, bias=True)
        self.fuse = nn.Conv2d(self.channel // m, planes // 2, 1, padding=0, bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        #self.prelu = nn.PReLU() 

    def forward(self, x):
        N, C, H, W = x.shape
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
        # [N * 3^2 * C/m * 1]
        kernel_atrous = self.conv_kernel2(g_perm)
        # [N * 1 * C/m * 3^2]
        kernel_atrous = kernel_atrous.permute(0, 3, 2, 1)

        # [1 * C/m * H * W]
        f_list = torch.split(f, 1, 0)
        # f_list = torch.split(x, 1, 0)
        g_list = torch.split(kernel, 1, 0)
        g_list_atrous = torch.split(kernel_atrous, 1, 0)

        out = []
        out_atrous = []
        for i in range(N):
            # [1* C/m * H * W]
            f_one = f_list[i]
            # [1 * 1 * C/m * k^2]
            g_one = g_list[i]
            # Also consider following form [C/m * 1 * 1 * k^2]
            #g_one = g_list[i].permute(2, 0, 1, 3)
            g_one_atrous  = g_list_atrous[i]

            ## [1 * 1 * C/m * k^2], also consider use 3*3 conv_k 
            ##  to utilize relationship among each channel and kernel
            g_k = self.conv_k(g_one)
            #g_k = F.relu(g_k)
            ## [C/m * 1 * k * k]
            g_k = g_k.reshape(g_k.size(2), g_k.size(1), self.k, self.k)
            #g_k = g_one.reshape(g_one.size(2), g_one.size(1), self.k, self.k)
            
            g_k_atrous = self.conv_k2(g_one_atrous)
            #g_k_atrous = F.relu(g_k_atrous)
            ## [C/m * 1 * 3 * 3]
            g_k_atrous = g_k_atrous.reshape(g_k_atrous.size(2), g_k_atrous.size(1), 3, 3)
            #g_k_atrous = g_one_atrous.reshape(g_one_atrous.size(2), g_one_atrous.size(1), 3, 3)


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
        y = self.fuse(y)
        y_atrous = torch.cat(out_atrous, dim=0)
        y_atrous = self.fuse(y_atrous)
        # [N * 2C/m * H * W]
        y = torch.cat([y, y_atrous], dim=1)
        #y = self.fuse(y)
        return y

class DIDAModule(nn.Module):
    def __init__(self, k, inplanes, planes, m=4, padding=None, stride=1):
        super(DIDAModule, self).__init__()
        self.k = k
        self.channel = inplanes 
        self.group = self.channel // m
        self.conv = nn.Conv2d(self.channel, self.channel // m, 1, padding=0, bias=True)
        self.pad = padding
        self.stride = stride
        
        self.conv_k = nn.Conv2d(1, 1, 1, padding=0, bias=True)
        self.conv_k2 = nn.Conv2d(1, 1, 1, padding=0, bias=True)
        self.conv_kernel = nn.Conv2d(1, k*k, 1, padding=0, bias=True)
        self.conv_kernel2 = nn.Conv2d(1, 3*3, 1, padding=0, bias=True)
        self.fuse = nn.Conv2d(self.channel // m, planes // 2, 1, padding=0, bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        N, C, H, W = x.shape
        # [N * C/m * H * W]
        f = F.relu(self.conv(x))
        # [N * C/m * 1 * 1]
        g = F.relu(self.conv(self.avg_pool(x))) 
        # [N * 1 * C/m * 1]
        g_perm = g.permute(0, 2, 1, 3).contiguous()
        # [N * k^2 * C/m * 1]
        kernel = self.conv_kernel(g_perm)
        # [N * 1 * C/m * k^2]
        kernel = kernel.permute(0, 3, 2, 1)
        # [N * 3^2 * C/m * 1]
        kernel_atrous = self.conv_kernel2(g_perm)
        # [N * 1 * C/m * 3^2]
        kernel_atrous = kernel_atrous.permute(0, 3, 2, 1)

        # [1 * C/m * H * W]
        f_list = torch.split(f, 1, 0)
        g_list = torch.split(kernel, 1, 0)
        g_list_atrous = torch.split(kernel_atrous, 1, 0)

        out = []
        out_atrous = []
        for i in range(N):
            # [1* C/m * H * W]
            f_one = f_list[i]
            # [1 * 1 * C/m * k^2]
            g_one = g_list[i]
            g_one_atrous  = g_list_atrous[i]

            ## [1 * 1 * C/m * k^2]
            g_k = self.conv_k(g_one)
            ## [C/m * 1 * k * k]
            g_k = g_k.reshape(g_k.size(2), g_k.size(1), self.k, self.k)
            g_k_atrous = self.conv_k2(g_one_atrous)
            ## [C/m * 1 * 3 * 3]
            g_k_atrous = g_k_atrous.reshape(g_k_atrous.size(2), g_k_atrous.size(1), 3, 3)

            # [1* C/m * H * W]
            if self.pad is None:
                padding = ((self.k-1) // 2, (self.k-1) // 2, (self.k-1) // 2, (self.k-1) // 2)
            else:
                padding = (self.pad, self.pad, self.pad, self.pad)
            x = F.pad(input=f_one, pad=padding, mode='constant', value=0)
            o = F.conv2d(input=x, weight=g_k, stride=self.stride, groups=self.group)
            out.append(o)
            x = F.pad(input=f_one, pad=(2,2,2,2), mode='constant', value=0)
            o_atrous = F.conv2d(input=x, weight=g_k_atrous, stride=self.stride, dilation=2, groups=self.group)
            out_atrous.append(o_atrous)

        # [N * C/m * H * W]
        y = torch.cat(out, dim=0)
        y = self.fuse(y)
        y_atrous = torch.cat(out_atrous, dim=0)
        # [N * 2C/m * H * W]
        y_atrous = self.fuse(y_atrous)
        y = torch.cat([y, y_atrous], dim=1) 
        return y


class DIDAModuleD4(nn.Module):
    def __init__(self, k, inplanes, planes, m=4, padding=None, stride=1):
        super(DIDAModuleD4, self).__init__()
        self.k = k
        self.channel = inplanes 
        self.group = self.channel // m
        self.conv = nn.Conv2d(self.channel, self.channel // m, 1, padding=0, bias=True)
        self.pad = padding
        self.stride = stride
        #if padding is None:
        #    #self.conv_adap_kernel = nn.Conv2d(self.channel // m, self.channel // m, self.k, padding=(self.k-1) // 2, stride=stride)
        #    self.conv_adap_kernel = nn.Conv2d(self.channel // m, self.channel // m, self.k, padding=(self.k-1) // 2, stride=stride, groups=self.channel // m)
        #    self.conv3x3_atrous_adap_kernel = nn.Conv2d(self.channel // m, self.channel // m, 3, padding=2, stride=stride, dilation=2, groups=self.channel // m)
        #else:
        #    #self.conv_adap_kernel = nn.Conv2d(self.channel // m, self.channel // m, self.k, padding=padding, stride=stride)
        #    self.conv_adap_kernel = nn.Conv2d(self.channel // m, self.channel // m, self.k, padding=padding, stride=stride, groups=self.channel // m)
        #    self.conv3x3_atrous_adap_kernel = nn.Conv2d(self.channel // m, self.channel // m, 3, padding=2, stride=stride, dilation=2, groups=self.channel // m)

        #self.conv_k = nn.Conv2d(1, self.channel // m, 3, padding=1, bias=True)
        self.conv_k = nn.Conv2d(1, 1, 1, padding=0, bias=True)
        self.conv_k2 = nn.Conv2d(1, 1, 1, padding=0, bias=True)
        self.conv_k_d4 = nn.Conv2d(1, 1, 1, padding=0, bias=True)
        #self.conv_for_g = nn.Conv2d(self.channel, self.channel // m, 1, padding=0, bias=True)
        self.conv_kernel = nn.Conv2d(1, k*k, 1, padding=0, bias=True)
        self.conv_kernel2 = nn.Conv2d(1, 3*3, 1, padding=0, bias=True)
        self.conv_kernel_d4 = nn.Conv2d(1, 3*3, 1, padding=0, bias=True)
        self.fuse = nn.Conv2d(self.channel // m, planes // 3, 1, padding=0, bias=True)
        #self.fuse_atrous = nn.Conv2d(self.channel // m, planes // 2, 1, padding=0, bias=True)
        self.fuse_channel = nn.Conv2d(planes // 3 * 3, planes, 1, padding=0, bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #self.max_pool = nn.AdaptiveMaxPool2d(1)
        #self.prelu = nn.PReLU() 

    def forward(self, x):
        N, C, H, W = x.shape
        # [N * C/m * H * W]
        f = F.relu(self.conv(x))
        # [N * C/m * 1 * 1]
        g = F.relu(self.conv(self.avg_pool(x))) #+ F.relu(self.conv(self.max_pool(x)))
        #pdb.set_trace()
        # [N * 1 * C/m * 1]
        g_perm = g.permute(0, 2, 1, 3).contiguous()
        # [N * k^2 * C/m * 1]
        kernel = self.conv_kernel(g_perm)
        #kernel = F.relu(kernel)
        # [N * 1 * C/m * k^2]
        kernel = kernel.permute(0, 3, 2, 1)
        #g = F.relu(self.conv(pooled_feat))
        # [N * 3^2 * C/m * 1]
        kernel_atrous = self.conv_kernel2(g_perm)
        # [N * 1 * C/m * 3^2]
        kernel_atrous = kernel_atrous.permute(0, 3, 2, 1)
        
        k_atrous_d4 = self.conv_kernel_d4(g_perm)
        k_atrous_d4 = k_atrous_d4.permute(0, 3, 2, 1)

        # [1 * C/m * H * W]
        f_list = torch.split(f, 1, 0)
        # f_list = torch.split(x, 1, 0)
        g_list = torch.split(kernel, 1, 0)
        g_list_atrous = torch.split(kernel_atrous, 1, 0)
        g_list_atrous_d4 = torch.split(k_atrous_d4, 1, 0)

        out = []
        out_atrous = []
        out_atrous_d4 = []
        for i in range(N):
            # [1* C/m * H * W]
            f_one = f_list[i]
            # [1 * 1 * C/m * k^2]
            g_one = g_list[i]
            # Also consider following form [C/m * 1 * 1 * k^2]
            #g_one = g_list[i].permute(2, 0, 1, 3)
            g_one_atrous  = g_list_atrous[i]
            g_one_atrous_d4 = g_list_atrous_d4[i]

            ## [1 * 1 * C/m * k^2], also consider use 3*3 conv_k 
            ##  to utilize relationship among each channel and kernel
            g_k = self.conv_k(g_one)
            #g_k = F.relu(g_k)
            ## [C/m * 1 * k * k]
            g_k = g_k.reshape(g_k.size(2), g_k.size(1), self.k, self.k)
            #g_k = g_one.reshape(g_one.size(2), g_one.size(1), self.k, self.k)
            g_k_atrous = self.conv_k2(g_one_atrous)
            #g_k_atrous = F.relu(g_k_atrous)
            ## [C/m * 1 * 3 * 3]
            g_k_atrous = g_k_atrous.reshape(g_k_atrous.size(2), g_k_atrous.size(1), 3, 3)
            #g_k_atrous = g_one_atrous.reshape(g_one_atrous.size(2), g_one_atrous.size(1), 3, 3)

            g_k_atrous_d4 = self.conv_k_d4(g_one_atrous_d4)
            g_k_atrous_d4 = g_k_atrous_d4.reshape(g_k_atrous_d4.size(2), g_k_atrous_d4.size(1), 3, 3)

            # [1* C/m * H * W]
            #o = self.conv_adap_kernel(f_one)
            if self.pad is None:
                padding = ((self.k-1) // 2, (self.k-1) // 2, (self.k-1) // 2, (self.k-1) // 2)
            else:
                padding = (self.pad, self.pad, self.pad, self.pad)
            x = F.pad(input=f_one, pad=padding, mode='constant', value=0)
            o = F.conv2d(input=x, weight=g_k, stride=self.stride, groups=self.group)
            #print('o shape {}, f shape {}'.format(o.shape, f_one.shape))
            out.append(o)
            x = F.pad(input=f_one, pad=(2,2,2,2), mode='constant', value=0)
            #o_atrous = self.conv3x3_atrous_adap_kernel(f_one)
            o_atrous = F.conv2d(input=x, weight=g_k_atrous, stride=self.stride, dilation=2, groups=self.group)
            out_atrous.append(o_atrous)
            
            x = F.pad(input=f_one, pad=(4,4,4,4), mode='constant', value=0)
            o_atrous_d4 = F.conv2d(input=x, weight=g_k_atrous_d4, stride=self.stride, dilation=4, groups=self.group)
            out_atrous_d4.append(o_atrous_d4)

        # [N * C/m * H * W]
        y = torch.cat(out, dim=0)
        #y = F.relu(y)
        y = self.fuse(y)
        y_atrous = torch.cat(out_atrous, dim=0)
        # [N * 2C/m * H * W]
        y_atrous = self.fuse(y_atrous)
        
        y_atrous_d4 = torch.cat(out_atrous_d4, dim=0)
        y_atrous_d4 = self.fuse(y_atrous_d4)
        y = torch.cat([y, y_atrous, y_atrous_d4], dim=1) #y + y_atrous 
        return self.fuse_channel(y)


class GlobalMultiScaleAdaptKernelBaseline(nn.Module):
    def __init__(self, k, inplanes, planes, m=4, padding=None, stride=1):
        super(GlobalMultiScaleAdaptKernelBaseline, self).__init__()
        self.k = k
        self.channel = inplanes
        self.conv = nn.Conv2d(self.channel, self.channel // m, 1, padding=0, bias=True)
        #self.conv_reduce1 = nn.Conv2d(self.channel, self.channel // m, 1, padding=0, bias=True)
        #self.conv_reduce2 = nn.Conv2d(self.channel, self.channel // m, 1, padding=0, bias=True)
        self.bn1 = nn.BatchNorm2d(self.channel // m)
        if padding is None:
            #self.conv_adap_kernel = nn.Conv2d(self.channel // m, self.channel // m, self.k, padding=(self.k-1) // 2, stride=stride)
            self.conv_adap_kernel = nn.Conv2d(self.channel // m, self.channel // m, self.k, padding=(self.k-1) // 2, stride=stride, groups=self.channel // m)
            #self.conv3x3_atrous_adap_kernel = nn.Conv2d(self.channel // m, self.channel // m, 3, padding=2, stride=stride, dilation=2, groups=self.channel // m)
            self.conv1x1_adap_kernel = nn.Conv2d(self.channel // m, self.channel // m, 1, padding=0, stride=stride, groups=self.channel // m)
        else:
            #self.conv_adap_kernel = nn.Conv2d(self.channel // m, self.channel // m, self.k, padding=padding, stride=stride)
            self.conv_adap_kernel = nn.Conv2d(self.channel // m, self.channel // m, self.k, padding=padding, stride=stride, groups=self.channel // m)
            #self.conv3x3_atrous_adap_kernel = nn.Conv2d(self.channel // m, self.channel // m, 3, padding=2, stride=stride, dilation=2, groups=self.channel // m)
            self.conv1x1_adap_kernel = nn.Conv2d(self.channel // m, self.channel // m, 1, padding=0, stride=stride, groups=self.channel // m)
        self.bn2 = nn.BatchNorm2d(self.channel // m)
        #self.conv_k = nn.Conv2d(1, self.channel // m, 3, padding=1, bias=True)
        self.conv_k = nn.Conv2d(1, 1, 1, padding=0, bias=True)
        self.conv_k2 = nn.Conv2d(1, 1, 1, padding=0, bias=True)
        #self.conv_for_g = nn.Conv2d(self.channel, self.channel // m, 1, padding=0, bias=True)
        self.conv_kernel = nn.Conv2d(1, k*k, 1, padding=0, bias=True)
        self.conv_kernel2 = nn.Conv2d(1, 1, 1, padding=0, bias=True)
        self.fuse = nn.Conv2d(self.channel // m, planes // 2, 1, padding=0, bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        #self.prelu = nn.PReLU() 

    def forward(self, x):
        N, C, H, W = x.shape
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
        # [N * 3^2 * C/m * 1]
        #g2 = F.relu(self.conv_reduce2(self.avg_pool(x))) 
        #g_perm2 = g2.permute(0, 2, 1, 3)
        kernel_atrous = self.conv_kernel2(g_perm)
        # [N * 1 * C/m * 3^2]
        kernel_atrous = kernel_atrous.permute(0, 3, 2, 1)

        # [1 * C/m * H * W]
        f_list = torch.split(f, 1, 0)
        # f_list = torch.split(x, 1, 0)
        g_list = torch.split(kernel, 1, 0)
        g_list_atrous = torch.split(kernel_atrous, 1, 0)

        out = []
        out_atrous = []
        for i in range(N):
            # [1* C/m * H * W]
            f_one = f_list[i]
            # [1 * 1 * C/m * k^2]
            g_one = g_list[i]
            # Also consider following form [C/m * 1 * 1 * k^2]
            #g_one = g_list[i].permute(2, 0, 1, 3)
            g_one_atrous  = g_list_atrous[i]

            ## [1 * 1 * C/m * k^2], also consider use 3*3 conv_k 
            ##  to utilize relationship among each channel and kernel
            g_k = self.conv_k(g_one)
            #g_k = F.relu(g_k)
            ## [C/m * 1 * k * k]
            g_k = g_k.reshape(g_k.size(2), g_k.size(1), self.k, self.k)
            #g_k = g_one.reshape(g_one.size(2), g_one.size(1), self.k, self.k)
            
            g_k_atrous = self.conv_k2(g_one_atrous)
            #g_k_atrous = F.relu(g_k_atrous)
            ## [C/m * 1 * 3 * 3]
            g_k_atrous = g_k_atrous.reshape(g_k_atrous.size(2), g_k_atrous.size(1), 1, 1)
            #g_k_atrous = g_one_atrous.reshape(g_one_atrous.size(2), g_one_atrous.size(1), 3, 3)


            self.conv_adap_kernel.weight = nn.Parameter(g_k)
            self.conv1x1_adap_kernel.weight = nn.Parameter(g_k_atrous)

            # [1* C/m * H * W]
            o = self.conv_adap_kernel(f_one)
            # print('o shape {}'.format(o.shape))
            out.append(o)
            o_atrous = self.conv1x1_adap_kernel(f_one)
            out_atrous.append(o_atrous)

        # [N * C/m * H * W]
        y = torch.cat(out, dim=0)
        #y = F.relu(y)
        y = self.fuse(y)
        y_atrous = torch.cat(out_atrous, dim=0)
        y_atrous = self.fuse(y_atrous)
        # [N * 2C/m * H * W]
        y = torch.cat([y, y_atrous], dim=1)
        #y = self.fuse(y)
        return y

