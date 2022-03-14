# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 10:55:41 2021

@author: rw17789
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from cross_attention_network import Cross_Attention, Self_Attention

from torch.nn.utils import spectral_norm

from utils import median_calculater, mean_calculater

def snconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
    return spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, 
                                   dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode))

def snconvtranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros'):
    return spectral_norm(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, 
                                            groups=groups, bias=bias, dilation=dilation, padding_mode=padding_mode))

class GenUnit(nn.Module):
    def __init__(self, in_channels, out_channels, dtype, activation = 'relu', use_dropout = False, use_sn = True):
        super(GenUnit, self).__init__()
        assert dtype in ['down', 'up', 'in']
        assert activation in ['relu', 'leaky']
        
        if dtype in ['down', 'in']:
            conv2d = snconv2d if use_sn else nn.Conv2d
        else:
            conv2d = snconvtranspose2d if use_sn else nn.ConvTranspose2d
            
        layerlist = []
        
        if dtype == "down":
            layerlist.append(
                conv2d(in_channels, out_channels, 4, 2, 1, bias = False, padding_mode='reflect')
            )
        elif dtype == "in":
            layerlist.append(
                conv2d(in_channels, out_channels, 3, 1, 1, bias = False, padding_mode='reflect')
            )
        else:
            layerlist.append(
                conv2d(in_channels, out_channels, 4, 2, 1, bias = False)
            )
            
        # if not use_sn:
        layerlist.append(nn.BatchNorm2d(out_channels))
        
        if use_dropout:
            layerlist.append(nn.Dropout(0.5))
            
        layerlist.append(
            nn.ReLU() if activation =='relu' else nn.LeakyReLU(0.2)
            )
        
        self.unit = nn.Sequential(*layerlist)
        
    def forward(self, x):
        return self.unit(x)
    
class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dtype, activation = 'relu', use_dropout = False, add_Attention = False, use_sn = True, middel_channels = None):
        super(GenBlock, self).__init__()
        assert dtype in ['down', 'up'], 'Block type should be down or up'
        assert activation in ['relu', 'leaky']
        
        conv2d = snconv2d if use_sn else nn.Conv2d
        
        layerlist = []
        
        self.dtype = dtype
        
        if dtype == "down":
            self.block = nn.Sequential(
                conv2d(in_channels, out_channels, 1, 1, bias = False),
                nn.MaxPool2d(kernel_size = 2, stride = 2)
                )
            # layerlist.append(
            #     GenUnit(in_channels, in_channels, "in", activation, use_dropout, use_sn)
            #     )
            layerlist.append(
                GenUnit(in_channels, in_channels if middel_channels is None else middel_channels, "in", activation, use_dropout, use_sn)
                )
            if middel_channels is not None:
                in_channels = middel_channels
        layerlist.append(
            GenUnit(in_channels, out_channels, dtype, activation, use_dropout, use_sn)
            )
        if dtype == "up":
            self.block = nn.Sequential(
                conv2d(in_channels, out_channels, 1, 1, bias = False),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                )
            layerlist.append(
                GenUnit(out_channels, out_channels, "in", activation, use_dropout, use_sn)
                )
        
        self.residual = nn.Sequential(*layerlist)
        
        self.attention = Self_Attention(in_channels if dtype == "down" else out_channels, use_sn = use_sn) if add_Attention else None
                
    def forward(self, x):
        if self.attention is None:
            return self.block(x) + self.residual(x)
        elif self.dtype == "down":
            x = self.attention(x)
            return self.block(x) + self.residual(x)
        else:
            return self.attention(self.block(x) + self.residual(x))
        
class Generator(nn.Module):
    """
    Generator
    input feature maps size shouble be 256
    """
    def __init__(self, in_channels, out_channels, attn_pos='01110', has_mask = False, hidden_channels = 64, down_step = 5, bypass = False, use_sn = True):
        super(Generator, self).__init__()
        assert down_step in [4,5]
        assert len(attn_pos) >= down_step
        pos = [p=="1" for p in attn_pos[-down_step:]]
        
        conv2d = snconv2d if use_sn else nn.Conv2d
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.has_mask = has_mask
        
        self.down = nn.ModuleList(
            [
                GenBlock(
                    in_channels if i == 0 else hidden_channels * min(2**(i-1), 8), 
                    hidden_channels * min(2**i, 8), 
                    'down', 
                    'leaky', 
                    False, 
                    False, 
                    use_sn, 
                    hidden_channels if i==0 else None
                    )
                for i in range(down_step)] 
            )

        self.midblock = nn.Sequential(
            GenUnit(hidden_channels*8, hidden_channels*8, 'in', 'relu', False, use_sn),
            GenUnit(hidden_channels*8, hidden_channels*8, 'in', 'relu', False, use_sn)
            )
        
        self.up = nn.ModuleList(
            [
                GenBlock(
                    hidden_channels * 8 if i==0 else hidden_channels * (2 ** (down_step - i)),
                    hidden_channels * (2 ** max(down_step - 2 -i, 0)),
                    'up',
                    'relu',
                    i < (down_step - 2),
                    False if self.has_mask else p,
                    use_sn,
                    None
                    )
                for i,p in enumerate(pos)]
            )
        
        if self.has_mask:
            self.up_mask = nn.ModuleList(
                [
                    GenBlock(
                        hidden_channels * 8 if i==0 else hidden_channels * (2 ** (down_step - i)),
                        hidden_channels * (2 ** max(down_step - 2 -i, 0)),
                        'up',
                        'relu',
                        i < (down_step - 2),
                        False,
                        use_sn,
                        None
                        )
                    for i in range(down_step)] 
                )
            self.cross_attention = nn.ModuleList(
                [
                    Cross_Attention(
                        hidden_channels * (2 ** max(down_step - 2 -i, 0)),
                        bypass,
                        False
                        ) if p else nn.Sequential() for i, p in enumerate(pos)
                    ]
                )      
        
        self.final = nn.Sequential(
            conv2d(hidden_channels, out_channels, 3, 1, 1, padding_mode='reflect'),
            nn.Tanh()
            )
        if self.has_mask:
            self.final_mask = conv2d(hidden_channels, out_channels + 1, 3, 1, 1, padding_mode='reflect')
            # self.final_mask = nn.ModuleList(
            #     [conv2d(hidden_channels, 2, 3, 1, 1, padding_mode='reflect') for _ in range(out_channels)]
            #     )
        
    def forward(self, x):
        d = []
        for layer in self.down:
            d.append(layer(d[-1] if len(d) != 0 else x))

        mid = self.midblock(d[-1]) + d[-1]
        
        if self.has_mask:
            for i, (layer, layer_mask, cross_layer) in enumerate(zip(self.up, self.up_mask, self.cross_attention)):
                p = layer(mid if i == 0 else torch.cat([p, d[-i-1]], dim=1))
                m = layer_mask(mid if i == 0 else torch.cat([m, d[-i-1]], dim=1))
                # print(i,"(p):",p.shape)
                # print(i,"(m):",m.shape)
                if not isinstance(cross_layer, nn.Sequential):
                    if cross_layer.by_pass:
                        p = cross_layer(p,m)
                    else:
                        p, m = cross_layer(p,m)
                        # print(i,"(p*):",p.shape)
                        # print(i,"(m*):",m.shape)                        
        else:
            for i, layer in enumerate(self.up):
                p = layer(mid if i == 0 else torch.cat([p, d[-i-1]], dim=1))
        
        out = {"image": self.final(p)}

        if self.has_mask:
            mask = self.final_mask(m)
            out["mask"] = mask.unsqueeze(2)
            img_mask = F.softmax(mask, dim=1)[:,1:,...]
            # out["image"] = (out["image"] + 1) * img_mask - 1
            if self.training:
                out["image"] = (out["image"] + 1) * img_mask - 1
            else:
                out["image"] = (out["image"] + 1) * img_mask.round() - 1        
        return out

    def regularization(self, x):
        F.relu(x - 0.5) / (torch.abs(x - 0.5) + 1e-8)
    
class DisBlock(nn.Module):
    """
    Block for building Discriminator
    in_channels: number of input channels
    out_channels: number of output channels
    All Block contains Convolutional layers with kernel_size=4 and padding=1
    stride: stride for convlutional layers, can be 2(default) or 1
    has_bn: add BatchNormalization
    has_relu: add leaky RelU(0.2, True)
    use_sn: whether to use spectral normalization
    """
    def __init__(self, in_channels, out_channels, bias = False, add_Attention = False, use_sn = True):
        super(DisBlock, self).__init__()
        conv2d = snconv2d if use_sn else nn.Conv2d
        
        self.residual = nn.Sequential(
            conv2d(in_channels, out_channels, 3, 1, 1, bias = bias, padding_mode = 'reflect'),
            nn.ReLU()
            )
        self.down = nn.Sequential(
            conv2d(out_channels, out_channels, 4, 2, 1, bias = bias, padding_mode = 'reflect'),
            nn.LeakyReLU(0.2, True)
            )
        self.conv = nn.Sequential(
            conv2d(in_channels, out_channels, 1, 1, bias = bias),
            nn.MaxPool2d(2, 2)
            )
        self.attention = Self_Attention(out_channels, use_sn = use_sn) if add_Attention else None
        
    def forward(self, x):
        x_r = self.down(self.residual(x))
        x0 = self.conv(x)
        out = x_r + x0
        if self.attention is not None:
            return self.attention(out)
        else:
            return out
        
class Discriminator(nn.Module):
    def __init__(self, in_channels, hidden_channels = 64, attention_level= "Non", down_step = 5, use_sn = True, batch = True):
        super(Discriminator, self).__init__()
        assert down_step in [4,5]
        assert attention_level in [ 'non', 'Non', 'Top', 'top', 'Sec', 'sec', 'Two', 'two']
        pos = [False, False]
        if attention_level in ['Top', 'top', 'Two', 'two']:
            pos[0] = True 
        if attention_level in ['Sec', 'sec', 'Two', 'two']:
            pos[1] = True
            
        self.batch = batch
        
        self.layer1 = nn.Sequential(
            DisBlock(in_channels, hidden_channels, True, pos[0], use_sn),
            DisBlock(hidden_channels, hidden_channels * 2, False, pos[1], use_sn)
            )
        
        if self.batch:
            conv2d = snconv2d if use_sn else nn.Conv2d
            self.layer2 = nn.Sequential(
                conv2d(hidden_channels * 2, hidden_channels * 4, 4, 1, 1, bias = False, padding_mode = 'reflect'),
                nn.ReLU(),
                conv2d(hidden_channels * 4, 1, 4, 1, 1, bias = False, padding_mode = 'reflect'),
                nn.ReLU()
                )
            self.direct = conv2d(hidden_channels * 2, 1, 1, bias = False)
        else:
            self.layer2 = nn.Sequential(
                DisBlock(hidden_channels * 2, hidden_channels * 4, True, False, use_sn),
                DisBlock(hidden_channels * 4, hidden_channels * 8, False, False, use_sn),
                DisBlock(hidden_channels * 8, 1, False, False, use_sn)
                )
        
    def forward(self, x, y=None):
        if y is None:
            input = x
        else:
            assert (x.size(2)==y.size(2)) and (x.size(3)==y.size(3)), 'image shape should be the same'
            input = torch.cat([x,y],1)
        
        out1 = self.layer1(input)
        out2 = self.layer2(out1)
        if self.batch:
            out = out2 + self.direct(out1)[..., 1:-1, 1:-1]
            return out
        else:
            out = out2.sum(dim=(2,3)).squeeze()
            return out
