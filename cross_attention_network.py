# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 18:34:05 2021

@author: rw17789
"""

import torch
import torch.nn as nn

from torch.nn.utils import spectral_norm

def snconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
    return spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, 
                                   dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode))

def snconvtranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros'):
    return spectral_norm(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, 
                                            groups=groups, bias=bias, dilation=dilation, padding_mode=padding_mode))

class Self_Attention(nn.Module):
    """
    Self_Attention network based on Self-Attention Generative Adversarial Networks
    
    in_channels: number of channels of input
    hidden_scale: k in the paper, reduce number of channel to Ch/k, where k = 1,2,4,8, defaut 8
    use_sn: whether use Spectral Normalization, default True

    """
    def __init__(self, in_channels, hidden_scale = 8, use_sn = True):
        super(Self_Attention, self).__init__()
        
        conv2d = snconv2d if use_sn else nn.Conv2d

        assert in_channels//hidden_scale > 1
        self.in_channels = in_channels
        
        self.conv_f = conv2d(in_channels = in_channels, out_channels = in_channels//hidden_scale, kernel_size = 1)
        self.conv_g = conv2d(in_channels = in_channels, out_channels = in_channels//hidden_scale, kernel_size = 1)
        self.conv_h = conv2d(in_channels = in_channels, out_channels = in_channels//hidden_scale, kernel_size = 1)
        self.conv_v = conv2d(in_channels = in_channels//hidden_scale, out_channels = in_channels, kernel_size = 1)
        
        self.softmax = nn.Softmax(dim = -1)
        self.sigma = nn.Parameter(torch.zeros(1))
            
    def forward(self, x):
        """
        Parameters
        ----------
        x : input feature maps, size Bs * Ch * H *W

        Returns
        -------
        residual output: size Bs * Ch * H * W
        
        """       
        bs, _, H, W = x.size()
        
        f = self.conv_f(x).view(bs, -1, H*W)
        
        g = self.conv_g(x).view(bs, -1, H*W)

        attn = self.softmax(torch.bmm(f.permute(0,2,1),g))
        
        h = self.conv_h(x).view(bs, -1, H*W)
        
        v = self.conv_v(torch.bmm(h,attn.permute(0,2,1)).view(bs, -1, H, W))
    
        return x + self.sigma * v
    
class Cross_Attention_Bypass(nn.Module):
    def __init__(self, in_channels, hidden_scale = 8, use_sn = True):
        super(Cross_Attention_Bypass, self).__init__()
        
        conv2d = snconv2d if use_sn else nn.Conv2d

        assert in_channels//hidden_scale > 1
        self.in_channels = in_channels
        
        self.conv_f_1 = conv2d(in_channels = in_channels, out_channels = in_channels//hidden_scale, kernel_size = 1)
        self.conv_g_1 = conv2d(in_channels = in_channels, out_channels = in_channels//hidden_scale, kernel_size = 1)
        self.conv_h_1 = conv2d(in_channels = in_channels, out_channels = in_channels//hidden_scale, kernel_size = 1)
        self.conv_v_11 = conv2d(in_channels = in_channels//hidden_scale, out_channels = in_channels, kernel_size = 1)
        self.conv_v_12 = conv2d(in_channels = in_channels//hidden_scale, out_channels = in_channels, kernel_size = 1)
        
        # self.conv_f_2 = conv2d(in_channels = in_channels, out_channels = in_channels//hidden_scale, kernel_size = 1)
        self.conv_g_2 = conv2d(in_channels = in_channels, out_channels = in_channels//hidden_scale, kernel_size = 1)
        self.conv_h_2 = conv2d(in_channels = in_channels, out_channels = in_channels//hidden_scale, kernel_size = 1)
        # if self.bilateral:
        #     self.conv_v_21 = conv2d(in_channels = in_channels//hidden_scale, out_channels = in_channels, kernel_size = 1)
        # self.conv_v_22 = conv2d(in_channels = in_channels//hidden_scale, out_channels = in_channels, kernel_size = 1)
        
        self.softmax = nn.Softmax(dim = -1)
        
        self.conv_o_1 = conv2d(in_channels = in_channels * 2, out_channels = in_channels, kernel_size = 1)
        # if self.bilateral:
        #     self.conv_o_2 = conv2d(in_channels = in_channels * 2, out_channels = in_channels, kernel_size = 1)
        # else:
        #     self.sigma = nn.Parameter(torch.zeros(1))
            
    def forward(self, x, y):
        assert x.size() == y.size()
        bs, _, H, W = x.size()
        
        f_1 = self.conv_f_1(x).view(bs, -1, H*W)        
        g_1 = self.conv_g_1(x).view(bs, -1, H*W)
        
        # f_2 = self.conv_f_2(y).view(bs, -1, H*W)        
        g_2 = self.conv_g_2(y).view(bs, -1, H*W)
        
        attn_11 = self.softmax(torch.bmm(f_1.permute(0,2,1),g_1))
        attn_12 = self.softmax(torch.bmm(f_1.permute(0,2,1),g_2))
        # if self.bilateral:
        #     attn_21 = self.softmax(torch.bmm(f_2.permute(0,2,1),g_1))
        # attn_22 = self.softmax(torch.bmm(f_2.permute(0,2,1),g_2))
        
        h_1 = self.conv_h_1(x).view(bs, -1, H*W)
        h_2 = self.conv_h_2(x).view(bs, -1, H*W)
        
        v_11 = self.conv_v_11(torch.bmm(h_1,attn_11.permute(0,2,1)).view(bs, -1, H, W))
        v_12 = self.conv_v_12(torch.bmm(h_2,attn_12.permute(0,2,1)).view(bs, -1, H, W))
        # if self.bilateral:
        #     v_21 = self.conv_v_21(torch.bmm(h_1,attn_21.permute(0,2,1)).view(bs, -1, H, W))
        # v_22 = self.conv_v_22(torch.bmm(h_2,attn_22.permute(0,2,1)).view(bs, -1, H, W))
        
        x = x + self.conv_o_1(torch.cat([v_11, v_12], dim = 1))
        # y = y + (self.conv_o_2(torch.cat([v_21, v_22], dim = 1)) if self.bilateral else self.sigma * v_22) 
        
        return x, y
    
class Cross_Attention_bilateral(nn.Module):
    def __init__(self, in_channels, hidden_scale = 8, use_sn = True,  bilateral = False):
        super(Cross_Attention_bilateral, self).__init__()
        
        conv2d = snconv2d if use_sn else nn.Conv2d 
        
        assert in_channels//hidden_scale > 1
        self.bilateral = bilateral
        
        self.conv_f_1 = conv2d(in_channels = in_channels, out_channels = in_channels//hidden_scale, kernel_size = 1)
        self.conv_g_1 = conv2d(in_channels = in_channels, out_channels = in_channels//hidden_scale, kernel_size = 1)
        self.conv_h_1 = conv2d(in_channels = in_channels, out_channels = in_channels//hidden_scale, kernel_size = 1)
        self.conv_v_11 = conv2d(in_channels = in_channels//hidden_scale, out_channels = in_channels, kernel_size = 1)
        self.conv_v_12 = conv2d(in_channels = in_channels//hidden_scale, out_channels = in_channels, kernel_size = 1)
        
        self.conv_f_2 = conv2d(in_channels = in_channels, out_channels = in_channels//hidden_scale, kernel_size = 1)
        self.conv_g_2 = conv2d(in_channels = in_channels, out_channels = in_channels//hidden_scale, kernel_size = 1)
        self.conv_h_2 = conv2d(in_channels = in_channels, out_channels = in_channels//hidden_scale, kernel_size = 1)
        if self.bilateral:
            self.conv_v_21 = conv2d(in_channels = in_channels//hidden_scale, out_channels = in_channels, kernel_size = 1)
        self.conv_v_22 = conv2d(in_channels = in_channels//hidden_scale, out_channels = in_channels, kernel_size = 1)
        
        self.softmax = nn.Softmax(dim = -1)
        
        self.conv_o_1 = conv2d(in_channels = in_channels * 2, out_channels = in_channels, kernel_size = 1)
        if self.bilateral:
            self.conv_o_2 = conv2d(in_channels = in_channels * 2, out_channels = in_channels, kernel_size = 1)
        else:
            self.sigma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x, y):
        assert x.size() == y.size()
        bs, _, H, W = x.size()
        
        f_1 = self.conv_f_1(x).view(bs, -1, H*W)        
        g_1 = self.conv_g_1(x).view(bs, -1, H*W)
        
        f_2 = self.conv_f_2(y).view(bs, -1, H*W)        
        g_2 = self.conv_g_2(y).view(bs, -1, H*W)
        
        attn_11 = self.softmax(torch.bmm(f_1.permute(0,2,1),g_1))
        attn_12 = self.softmax(torch.bmm(f_1.permute(0,2,1),g_2))
        if self.bilateral:
            attn_21 = self.softmax(torch.bmm(f_2.permute(0,2,1),g_1))
        attn_22 = self.softmax(torch.bmm(f_2.permute(0,2,1),g_2))
        
        h_1 = self.conv_h_1(x).view(bs, -1, H*W)
        h_2 = self.conv_h_2(x).view(bs, -1, H*W)
        
        v_11 = self.conv_v_11(torch.bmm(h_1,attn_11.permute(0,2,1)).view(bs, -1, H, W))
        v_12 = self.conv_v_12(torch.bmm(h_2,attn_12.permute(0,2,1)).view(bs, -1, H, W))
        if self.bilateral:
            v_21 = self.conv_v_21(torch.bmm(h_1,attn_21.permute(0,2,1)).view(bs, -1, H, W))
        v_22 = self.conv_v_22(torch.bmm(h_2,attn_22.permute(0,2,1)).view(bs, -1, H, W))
        
        x = x + self.conv_o_1(torch.cat([v_11, v_12], dim = 1))
        y = y + (self.conv_o_2(torch.cat([v_21, v_22], dim = 1)) if self.bilateral else self.sigma * v_22) 
        
        return x, y
        

class Cross_Attention(nn.Module):
    def __init__(self, in_channels, by_pass=False, bilateral = False, hidden_scale = 8, use_sn = True):
        super(Cross_Attention, self).__init__()
        
        conv2d = snconv2d if use_sn else nn.Conv2d 
        
        assert in_channels//hidden_scale > 1
        self.by_pass = by_pass
        self.bilateral = bilateral
        
        self.conv_f_1 = conv2d(in_channels = in_channels, out_channels = in_channels//hidden_scale, kernel_size = 1)
        self.conv_g_1 = conv2d(in_channels = in_channels, out_channels = in_channels//hidden_scale, kernel_size = 1)
        self.conv_h_1 = conv2d(in_channels = in_channels, out_channels = in_channels//hidden_scale, kernel_size = 1)
        self.conv_v_11 = conv2d(in_channels = in_channels//hidden_scale, out_channels = in_channels, kernel_size = 1)
        self.conv_v_12 = conv2d(in_channels = in_channels//hidden_scale, out_channels = in_channels, kernel_size = 1)
        
        if not by_pass:
            self.conv_f_2 = conv2d(in_channels = in_channels, out_channels = in_channels//hidden_scale, kernel_size = 1)
        self.conv_g_2 = conv2d(in_channels = in_channels, out_channels = in_channels//hidden_scale, kernel_size = 1)
        self.conv_h_2 = conv2d(in_channels = in_channels, out_channels = in_channels//hidden_scale, kernel_size = 1)
        if not by_pass:
            if self.bilateral:
                self.conv_v_21 = conv2d(in_channels = in_channels//hidden_scale, out_channels = in_channels, kernel_size = 1)
            self.conv_v_22 = conv2d(in_channels = in_channels//hidden_scale, out_channels = in_channels, kernel_size = 1)
        
        self.softmax = nn.Softmax(dim = -1)
        
        self.conv_o_1 = conv2d(in_channels = in_channels * 2, out_channels = in_channels, kernel_size = 1)
        if not by_pass:
            if self.bilateral:
                self.conv_o_2 = conv2d(in_channels = in_channels * 2, out_channels = in_channels, kernel_size = 1)
            else:
                self.sigma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x, y):
        assert x.size() == y.size()
        bs, _, H, W = x.size()
        
        f_1 = self.conv_f_1(x).view(bs, -1, H*W)        
        g_1 = self.conv_g_1(x).view(bs, -1, H*W)
        
        if not self.by_pass:
            f_2 = self.conv_f_2(y).view(bs, -1, H*W)        
        g_2 = self.conv_g_2(y).view(bs, -1, H*W)
        
        attn_11 = self.softmax(torch.bmm(f_1.permute(0,2,1),g_1))
        attn_12 = self.softmax(torch.bmm(f_1.permute(0,2,1),g_2))
        if not self.by_pass:
            if self.bilateral:
                attn_21 = self.softmax(torch.bmm(f_2.permute(0,2,1),g_1))
            attn_22 = self.softmax(torch.bmm(f_2.permute(0,2,1),g_2))
        
        h_1 = self.conv_h_1(x).view(bs, -1, H*W)
        h_2 = self.conv_h_2(x).view(bs, -1, H*W)
        
        v_11 = self.conv_v_11(torch.bmm(h_1,attn_11.permute(0,2,1)).view(bs, -1, H, W))
        v_12 = self.conv_v_12(torch.bmm(h_2,attn_12.permute(0,2,1)).view(bs, -1, H, W))
        if not self.by_pass:
            if self.bilateral:
                v_21 = self.conv_v_21(torch.bmm(h_1,attn_21.permute(0,2,1)).view(bs, -1, H, W))
            v_22 = self.conv_v_22(torch.bmm(h_2,attn_22.permute(0,2,1)).view(bs, -1, H, W))
        
        x = x + self.conv_o_1(torch.cat([v_11, v_12], dim = 1))
        if self.by_pass:
            return x
        else:
            y = y + (self.conv_o_2(torch.cat([v_21, v_22], dim = 1)) if self.bilateral else self.sigma * v_22) 
            return x, y     
        
        
        
        
        
        