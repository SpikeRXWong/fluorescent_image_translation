# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 23:05:01 2021

@author: rw17789
"""

import os
import numpy as np
import cv2
from torch.utils.data import Dataset

from albumentations import (Flip, Compose, RandomRotate90, RandomResizedCrop, Rotate)
try:
    from albumentations.pytorch import ToTensor
except ImportError:
    from albumentations.pytorch import ToTensorV2 as ToTensor

def get_transforms(img_size, scale_range, version):
    """
    operation applied on dataset for augmentation 

    Parameters
    ----------
    img_size : int
        output image size.
    scale_range : float
        square value of the ratio of crop image to raw image.
    version : str
        purpose of the dataset.

    Returns
    -------
    list_trfms : albumentations.Compose
        augmentation operation on image.

    """
    assert img_size in [128, 256]
    
    item_list = []
    
    if version != "test":
        item_list.append(Flip())
        
    if version == "train":
        item_list.append(Rotate())
    elif version == "eval":
        item_list.append(RandomRotate90())
    
    item_list.append(RandomResizedCrop(img_size, img_size, scale = scale_range, ratio = (1, 1), p = 1))
    item_list.append(ToTensor())
    
    list_trfms = Compose(item_list, additional_targets={'image0': 'image', 'image1':'image'})

    return list_trfms  

class DBCImageDataset(Dataset):
    def __init__(self, path = None, output_type = 1, num_slice = 13, has_mask = True, img_size = 128, scale_range = None, version = "test"):
        """
        Dataset preparasion

        Parameters
        ----------
        path : str
            dataset store path.
        output_type : int, optional
            number of output channels of fluorescent images. The default is 1.
        num_slice : int, optional
            number of image slices of bright-field image stack. The default is 13.
        has_mask : bool, optional
            whether has mask generation path for the training model. The default is True.
        img_size : int, optional
            image width and height. The default is 128.
        scale_range : float, optional
            square value of ratio of the image size used in the model to size of raw images. If None ratio will be set to 0.25.
        version : str, optional
            purpose of dataset. The default is "test".
            
        Returns
        -------
        out : torch.utils.data.Dataset
            if has_mask is False, output brightfiled images and fluorescent images.
            else, output brightfiled images, fluorescent images, mask (classification), weights (calculated from mask used for EntropyCrossLoss calculation)

        """
        if path is None:
            path = 'DBCellfolder'
        self.path = path
        
        assert output_type in [1, 2]
        self.output_type = output_type
        # self.fluorescent = "fluorescent_{}".format(output_type)
        
        self.has_mask = has_mask
        
        assert img_size in [128, 256]
        
        assert num_slice <= 13   
        self.num = num_slice
        
        if scale_range is None:
            scale_range = ((img_size*2/512)**2,)*2

        assert version in ['train', 'eval', 'test']
        self.transform = get_transforms(img_size, scale_range, version)
    
    def __getitem__(self, idx):
        
        folder = os.path.join(self.path, os.listdir(self.path)[idx])
        
        fb = 0
        ff = 0
        if self.has_mask:
            nm = 0
        for n in os.listdir(folder):
            if 'brightfiled' in n:
                brightfield_image = self._resize(self.slice_select(np.load(os.path.join(folder, n)), self.num).transpose(1,2,0))
                fb += 1
            elif "fluorescent_{}".format(self.output_type) in n:
                if self.output_type == 1:
                    fluorescent_image = self._resize(np.expand_dims(np.load(os.path.join(folder, n)), axis = 2)) 
                else:
                    fluorescent_image = self._resize(np.load(os.path.join(folder, n))).transpose(1,2,0)
                ff += 1
            elif self.has_mask and ("mask" in n):
                mask_tuple = self._make_mask(np.load(os.path.join(folder, n)))
                nm += 1

        assert (fb == 1) and (ff == 1), "Bright_field or Fluorescent image not Found"
        if self.has_mask:
            assert nm == 1
            
        kwargs = {'image' : brightfield_image, 'image0' : fluorescent_image}
        if self.has_mask:
            kwargs['image1'] = mask_tuple
        
        aug = self.transform(**kwargs)
        
        brightfield_image = aug['image'].float()
        fluorescent_image = aug['image0'].float()
        out = {
            "brightfield": brightfield_image,
            "fluorescent": fluorescent_image
            }
        if self.has_mask:
            out["mask"] = aug['image1'][:1].long()
            out["weight"] = aug['image1'][1:].float()
        
        return out
    
    def slice_select(self, x, num):
        # keep image slack depth unchanged to be 3.6 μm (when slice is 1, select only the middle slice)
        assert num in list(range(1,14))
        ls = []
        if num == 1:
            ls.append(6)
        else:
            d = 12 / (num - 1)
            s = 0
            while s <= 12:
                ls.append(int(round(s)))
                s += d
        return x[ls, ...]
        # keep slice separation distance unchanged to be 0.3 μm
        # sta = 6 - num//2
        # end = sta + num 
        # # # x = x[sta : end]
        # return x[sta : end, ...]
    
    def __len__(self):
        return len(os.listdir(self.path))
    
    def _resize(self, x):
        x = x.astype(np.float32)
        return (x - x.min())/(x.max() - x.min()) * 2 -1
    
    def _single_weight_map(self, mask, r):
        maskT = (1 - mask)
        dist1 = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        dist2 = cv2.distanceTransform(maskT, cv2.DIST_L2, 5)
        
        dist1 = np.minimum(dist1, np.ones_like(dist1) * r)
        dist2 = np.minimum(dist2, np.ones_like(dist2) * r)
        
        weight = r - (dist1 + dist2)
        return weight
    
    def _make_mask(self, x):
        # x = x.astype(np.float32)
        mask = (x[0]*1 + x[1]*2).astype(np.float32)
        
        for i,m in enumerate(x):
            w = self._single_weight_map(m, 5)
            if i==0:
                weight = np.maximum(w, x[i].astype(np.float32) * 2)
            else:
                weight += np.maximum(w*3, x[i].astype(np.float32) * 4)
        weight = weight + 1
        
        return np.concatenate((np.expand_dims(mask, axis = 2), np.expand_dims(weight, axis = 2)), axis = 2)
        
