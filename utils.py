# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 16:37:40 2021

@author: rw17789
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def weight_def(epoch):
    # return 0.5 * (math.e**(-epoch/200))
    # return max(0.5 - (epoch//50)*0.1, 0)
    return 0

def Weighted_L1Loss(input, target, base = 1):
    base = base * math.e
    diff = (torch.abs(input - target))
    weight = torch.maximum(base ** target, math.e ** target)
    # weight = base ** (target + 1)
    return nn.functional.l1_loss(torch.mul(diff, weight), torch.zeros_like(target))

def Weighted_MSELoss(input, target, base = 1):
    base = math.e * base
    diff = torch.pow(input - target, 2)
    weight = torch.maximum(base ** target, math.e ** target)
    # weight = base ** (target + 1)
    return torch.nn.functional.l1_loss(torch.mul(diff, weight), torch.zeros_like(target))

def parameter_extractor(name):
    num_slice = int(name[name.index("_I")+2 : name.index("O",name.index("_I")+2)])
    out_slice = int(name[name.index("O", name.index("_I")+3)+1 : name.index("_G")])
    GAP = name[name.index("_G")+2 : name.index("_D")]
    DAL = name[name.index("_D")+2 : name.index("_S")]
    down_step = int(name[name.index("_S")+2 : name.index("_L")])
    loss_fl = name[name.index("_L")+2]
    return num_slice, out_slice, GAP, DAL, down_step, loss_fl

def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _compute_ssim(img1, img2, window, window_size, channel, R = 1, size_average = True):
    assert img1.dim() == 4
    assert img1.shape == img2.shape
    
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = (0.01 * R) ** 2
    C2 = (0.03 * R) ** 2
    

    output = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
    return output.mean() if size_average else output.mean(dim=(1,2,3))
    
def compute_ssim(img1, img2, range = (-1,1), window_size = 11, size_average = True):
    assert img1.dim() == 4
    assert img1.shape == img2.shape
    R = range[1] - range[0]
    img1 = img1 - range[0]
    img2 = img2 - range[0]
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _compute_ssim(img1, img2, window, window_size, channel, R, size_average)

def compute_pearson(x, y, size_average = True, value_range=(-1,1), threshold = False):
    """

    Parameters
    ----------
    x : float
        generated image
    y : flaat
        ground truth image.

    """
    assert x.shape == y.shape
    assert x.dim() == 4
    
    x_mean = x.mean(dim=(2,3), keepdim=True)
    y_mean = y.mean(dim=(2,3), keepdim=True)

    centered_x = x - x_mean
    centered_y = y - y_mean

    covariance = (centered_x * centered_y).mean(dim=(2,3), keepdim=True)

    x_std = x.std(dim=(2,3), keepdim=True)
    y_std = y.std(dim=(2,3), keepdim=True)

    corr = covariance / (x_std * y_std)
    
    if size_average and threshold:
        out = 0
        count = 0
        thres = (value_range[1] - value_range[0])*0.075 + value_range[0]
        bs, ch = x.shape[:2]
        for b in range(bs):
            for c in range(ch):
                if y_mean[b, c, 0, 0] > thres:
                    count += 1
                    out += corr[b, c, 0, 0]
        if count == 0:
            out = corr.mean()
            count = 1
        return out/count
    if size_average:
        return corr.mean() if not torch.isnan(corr.mean()) else -1
    else:
        return corr.mean(dim=(1,2,3))

# def compute_pearson(x, y, range=(-1,1), size_average = True):
#     assert x.dim() == 4
#     assert x.shape == y.shape
#     bs, ch, _, _ = x.shape
#     # assert ch == 1
#     if size_average:
#         count = 0
#         r_total = 0
#     r = torch.zeros(bs, ch).to(y.device)
#     for c in range(ch):    
#         for b in range(bs):
#             r[b, c] = _compute_pearson(x[b,c,...], y[b,c,...])
#             if size_average:
#                 if (x[b,c].max() >= (range[1] - range[0])*0.1 + range[0]) and (y[b,c].max() >= (range[1] - range[0])*0.1 + range[0]):
#                     count += 1
#                     r_total += r[b, c]
#     return r_total/count if size_average else r

# def _compute_pearson(x,y):
#     assert x.dim() == 2
#     assert x.shape == y.shape
#     numerator = torch.mul(x - x.mean(), y - y.mean()).sum()
#     x_dev = torch.square(x - x.mean()).sum()
#     y_dev = torch.square(y - y.mean()).sum()    
#     denominator = torch.sqrt(x_dev) * torch.sqrt(y_dev)
#     if denominator != 0:
#         return numerator / denominator
#     elif x_dev + y_dev == 0:
#         return torch.tensor(1.0)
#     else:
#         return torch.tensor(0.0)

def error_map_func(x, y, model = None, etype = "color", mode = "exhibit", criterion = None):
    
    assert x.dim() == 4
    assert y.dim() == 4
    assert etype in ["gray", "color"]
    assert mode in ["exhibit", "map", "error", "map&error"]
    
    if y.shape[1] != 1:
        etype = "gray"
    
    if ("error" in mode) and (criterion is None):
        criterion = nn.L1Loss()# if y.shape[1] == 1 else nn.MSELoss()
    
    if model is not None:
        model.eval()
        with torch.no_grad():
            x = model(x)
        
    assert y.shape == x.shape
    
    if mode != "error":
        if etype == "gray":
            diff = torch.abs(y - x).mean(dim=1, keepdim=True).repeat(1,3,1,1)
        else:
            diff_1 = torch.max(y - x, torch.zeros_like(y - x))
            diff_2 = torch.max(x - y, torch.zeros_like(x - y))
            diff = torch.cat([diff_1, diff_2, torch.zeros_like(y)], dim = 1)
        
    if mode == "exhibit":
        return diff - 1
    elif mode == "map&error":
        return criterion(x, y).item(), diff / 2
    elif mode == "map":
        return diff / 2
    else:
        return criterion(x, y).item()
    
def batch_intensity_compare(x, y):
    assert x.dim() == 4
    assert x.shape == y.shape
    assert x.shape[1] == 1
    image_size = x.shape[-1]
    bs = x.shape[0]
    result_list = []
    for b in range(bs):
        result = intensity_comparision(x[b,0,...], y[b,0,...])
        img_0 = turn2image(result, cell_wide = image_size//10 )
        image = torch.ones(3, image_size, image_size)
        edge = (image_size % 10) // 2
        image[:, edge:-edge, edge:-edge] = img_0
        result_list.append(image.unsqueeze(0))
    compare_result = torch.cat(result_list, dim = 0) * 2 - 1
    return compare_result.to(y.device)
  
def intensity_comparision(x,y, value_range = (-1,1)):
    assert x.dim() == 2
    assert x.shape == y.shape
    W, H  = x.shape
    result = torch.zeros(10,10)
    for i in range(W):
        for j in range(H):
            index_x = min(9, int((x[i,j] - value_range[0]) / (value_range[1] - value_range[0]) * 10))
            index_y = min(9, int((y[i,j] - value_range[0]) / (value_range[1] - value_range[0]) * 10))
            result[index_x, index_y] += 1
    return result

def max_index(x):
    index = torch.where(x == x.max())
    l = []
    for i in range(len(index[0])):
        ll = []
        for j in range(len(index)):
            ll.append(index[j][i].item())
        l.append(tuple(ll))
    return l

def turn2image(r, cell_wide = 10):
    assert r.dim() == 2
    W, H = r.shape
    max_loc = max_index(r)
    for p in max_loc:
        r[p] = 0
    r = r / (r.max() + 1e-7)
    r = torch.nn.Tanh()(r*5) * 0.9
    for p in max_loc:
        r[p] = 1
    p = torch.cat([torch.ones_like(r).unsqueeze(0), torch.zeros_like(r).unsqueeze(0), torch.ones_like(r).unsqueeze(0)], dim = 0)
    y = torch.cat([torch.ones_like(r).unsqueeze(0), torch.ones_like(r).unsqueeze(0), torch.zeros_like(r).unsqueeze(0)],dim = 0)
    map = (-torch.abs(2*r.unsqueeze(0)-1) + 1)*p + torch.max(2*r.unsqueeze(0)-1, torch.zeros_like(r.unsqueeze(0)))*y
    image = torch.zeros(3,W*cell_wide, H*cell_wide)
    for i in range(W):
        for j in range(H):
            for t in range(3):
                image[t, (9-i)*cell_wide: (10-i)*cell_wide, j*cell_wide: (j+1)*cell_wide] = map[t, i,j]
    return image

def float2long_transform(x, level=8, value_range = (-1,1)):
    # x = (x - value_range[0]) / (value_range[1] - value_range[0]) * level
    # x = torch.minimum(torch.floor(x), (level - 1) * torch.ones_like(x)).long()
    x = (x - value_range[0]) / (value_range[1] - value_range[0] + 2e-7) * level
    x = torch.floor(x).long()
    return x

def median_calculater(x, dim, soft_max = True, split = False, resize = True, value_range = (-1,1)):
    assert dim < x.dim()
    pm = (dim,) + tuple(range(dim)) + tuple(range(dim+1, x.dim()))
    # print(pm)
    x = x.permute(pm)
    if soft_max:
        x = nn.Softmax(dim=0)(x)
    # print(x.shape)
    base = torch.zeros(x.shape[1:]).float().to(x.device)
    # print(base.shape)
    acc = -torch.ones_like(base).int().to(x.device)
    total = torch.zeros_like(base).int().to(x.device)
    for i in range(x.shape[0]):
        base_new = base + x[i,...]
        layer = ((base <= 0.5) * (base_new > 0.5)).int()
        acc += layer * (i+1)
        base = base_new
        total += layer
    
    assert torch.equal(torch.round(base), torch.ones_like(base))
    assert torch.equal(total, torch.ones_like(total).int())
    
    if split:
        acc_list = []
        for i in range(1, x.shape[0]):
            acc_list.append((acc == i).float())
        acc = torch.cat(acc_list, dim = 1)
    
    if resize:
        acc = (acc / (x.shape[0] - 1)) * (value_range[1] - value_range[0]) + value_range[0]
        return acc.float()
    return acc

def mean_calculater(x, dim, soft_max = True, split = False, resize = True, value_range = (-1,1)):
    assert dim < x.dim()
    pm = (dim,) + tuple(range(dim)) + tuple(range(dim+1, x.dim()))
    # print(pm)
    x = x.permute(pm)
    if soft_max:
        x = nn.Softmax(dim=0)(x)
    # print(x.shape)
    N = x.shape[0]
    
    weight = torch.arange(N).view((-1,)+(1,)*(x.dim()-1)).repeat((1,)+tuple(x.shape[1:])).to(x.device)    
    if not split or N == 2:
        x = (weight * x).sum(dim = 0)
    else:
        x_list = []
        for i in range(1, N):
            w = (weight == i).float()
            x_list.append((w * x).sum(0))
        x = torch.cat(x_list, dim=1)
    
    if resize:
        x = (x / (N - 1)) * (value_range[1] - value_range[0]) + value_range[0]
        return x.float()
    return x.float()

def compute_psnr(img1, img2, img_range = None, reduction = "mean"):
    assert img1.shape == img2.shape, "Shape for inputs are different"
    assert reduction in ["mean", "none"]
    if img_range is None:
        img_range = torch.maximum(img1.max(), img2.max()) - torch.minimum(img1.min(), img2.min())
    mse = nn.functional.mse_loss(img1, img2, reduction='none').mean(dim = (1,2,3))
    psnr = 10 * torch.log10(img_range**2 / (mse + 1e-8))
    if reduction == "none":
        return psnr
    else:
        return psnr.mean()
