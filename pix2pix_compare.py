# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 15:43:38 2021

@author: rw17789
"""

import os
import time
# import sys
# import copy
import argparse


import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from pix2pix_model import Generator, Discriminator
from dbcelldataset import DBCImageDataset as ImageDataset

from train_tool_utils import train_one_epoch, eval_one_epoch, save_checkpoint, optimizer2device, save_example_image, save_model, test_model_performance

def parse_args():
    parser = argparse.ArgumentParser(description = "Parameter setting")
    
    parser.add_argument("-s", "--num_slice", default = 5, type = int, help="Number of slices of brightfield images no larger than 13")
        
    parser.add_argument("--out_slice", default = 1, type=int)
    
    parser.add_argument("-b", "--batch_size", default = 8, type=int, help="Batch size")
    
    parser.add_argument("-lg", "--lr_g", default = 1e-4, type = float)
    
    parser.add_argument("-ld", "--lr_d", default = 1e-4, type = float)
    
    parser.add_argument("-n","--num_epoches", default=300, type=int)
    
    parser.add_argument("--device", default="cuda", type=str, choices=["cpu", "cuda"])
    
    parser.add_argument("--RESUME", default=False, action="store_true")

    parser.add_argument("--LR_RESET", default=False, action="store_true")
    
    parser.add_argument("-c", "--checkpoint_file", type=str)
    
    parser.add_argument("--data_path", type=str) 
    
    parser.add_argument("--Destination", type=str)
    
    parser.add_argument("--image_size", default = 128, type = int, choices=[128, 256])
    
    parser.add_argument("--DS_scale", default = 5, type = int)
    
    parser.add_argument("--Lambda_1", default = 0.2, type = float)
    
    parser.add_argument("--Alpha", default = 0.8, type = float)
    
    parser.add_argument("--Lambda_2", default = 0.5, type = float)
    
    parser.add_argument("--Beta", default = 0.2, type = float)
       
    parser.add_argument("--image_interval", default = 20, type = int)
    
    parser.add_argument("--checkpoint_interval", default = 50, type = int)
    
    parser.add_argument("--Loss_Type", default = "DCGAN", type = str, choices=["DCGAN", "WGAN", "Hinge"])

    parser.add_argument("--test_path", type=str) 
    
    return parser.parse_args()


def main(args):
    
    start_time = time.time()
    start_time_point = time.localtime(start_time)
    
    print("Compare Pixel2Pixel model to Cross-Attention Conditional GAN for brightfield image to fluorescent image translation")

    print("Job ID : {}".format(os.environ['SLURM_JOBID']))
        
    print("Start time: {}/{:0>2d}/{:0>2d}\n".format(start_time_point.tm_year, start_time_point.tm_mon, start_time_point.tm_mday))
    

    if not torch.cuda.is_available():
        args.device = "cpu"
        
    args.has_mask = False
        
    if args.RESUME:
        assert args.checkpoint_file is not None
        
        checkpoint = torch.load(args.checkpoint_file, map_location = args.device)

        model_detail = checkpoint['Model_detail']
                
        start_epoch = checkpoint["Epoch"]
        if args.num_epoches <= checkpoint["Epoch"] + 1:
            args.num_epoches += checkpoint["Epoch"] + 1
    else:
        model_detail = "Pixel2Pixel_compare_model_{}_{:0>2d}{:0>2d}{:0>2d}{:0>2d}".format(os.environ['SLURM_JOBID'],
            start_time_point.tm_mon, start_time_point.tm_mday, start_time_point.tm_hour, start_time_point.tm_min
            )
        start_epoch = -1
    
    GEN = Generator(in_channels = args.num_slice, out_channels = args.out_slice)   
    GEN = GEN.to(args.device)
    DIS = {"Image": Discriminator(in_channels = args.num_slice + args.out_slice)}
    for key in DIS.keys():
        DIS[key] = DIS[key].to(args.device)
    
    optimizer_G = optim.Adam(GEN.parameters(), lr = args.lr_g, betas = (0.5, 0.999) if args.Loss_Type == "DCGAN" else (0.0, 0.9))

    optimizer_D = {"Image": optim.Adam(DIS["Image"].parameters(), lr = args.lr_d, betas = (0.5, 0.999) if args.Loss_Type == "DCGAN" else (0.0, 0.9))}
    
    if args.RESUME:
        GEN.load_state_dict(checkpoint["Generator_model"])
        DIS["Image"].load_state_dict(checkpoint["Discriminator_model"]["Image"])
        
        if not args.LR_RESET:
            optimizer_G.load_state_dict(checkpoint["Generator_optimizer"])
            optimizer_G = optimizer2device(optimizer_G, args.device)
                            
            optimizer_D["Image"].load_state_dict(checkpoint["Discriminator_optimizer"]["Image"])
            optimizer_D["Image"] = optimizer2device(optimizer_D["Image"], args.device)
            
    Folder_route = os.path.join(args.Destination, model_detail)
    if not os.path.exists(Folder_route):
        os.makedirs(Folder_route)
        
    if not args.RESUME:
        print("Start training of model {}\n".format(model_detail))
    else:
        print("Continue training of model {} from Epoch {}\n".format(model_detail, start_epoch+1))
        del checkpoint
    print("Model trained on {}\n".format(args.device))    
            
    org_train_dataset = ImageDataset(path = args.data_path, output_type = args.out_slice, num_slice = args.num_slice, has_mask = False, img_size = args.image_size, scale_range = None, version = "train")#tuple(args.scale_range[:2]) if len(args.scale_range)>1 else tuple(args.scale_range*2))
    org_eval_dataset = ImageDataset(path = args.data_path, output_type = args.out_slice, num_slice = args.num_slice, has_mask = False, img_size = args.image_size, scale_range = None, version = "eval")
    
    idx = max(int(len(org_train_dataset) / 20 + 0.1) * 4, 4)
    
    image_save_route = os.path.join(Folder_route, "Example_Images_{}".format(model_detail))
    if not os.path.exists(image_save_route):
        os.makedirs(image_save_route)
        
    checkpoint_folder = os.path.join(Folder_route, "Checkpoint_Storage_{}".format(model_detail))
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)
    
    loss_dict = {}
    loss_dict["Train_gen_loss"] = []
    loss_dict["Train_disc_loss"] = []
    loss_dict["Eval_gen_loss"] = []
    loss_dict["Eval_disc_loss"] = []
    loss_dict["Eval_Error"] = []
    # loss_dict["Pearson_r_value"] = []
    loss_dict["Distance"] = []
    loss_dict["PSNR"] = []
    
    for epoch in range(start_epoch+1, args.num_epoches):
        
        if (epoch % 500) == ((start_epoch+1) % 500):
            index = torch.randperm(len(org_train_dataset)).tolist()
            train_dataset = Subset(org_train_dataset, index[:-idx])
            eval_dataset = Subset(org_eval_dataset, index[-idx:])
            
            train_dataloader = DataLoader(
                train_dataset,
                batch_size = args.batch_size,
                shuffle = True,
                num_workers = 28 if args.device == 'cpu' else 0
            )
            eval_dataloader = DataLoader(
                eval_dataset,
                batch_size = 4,
                shuffle = False,
                num_workers = 4 if args.device == 'cpu' else 0
            )
        
        t0 = time.time()
        
        train_gen_loss, train_disc_loss = train_one_epoch(
            GEN, 
            DIS, 
            train_dataloader, 
            args.device, 
            epoch, 
            optimizer_G, 
            optimizer_D,
            args.DS_scale,# max(args.DS_scale - (epoch// 1000), 1),
            args.Lambda_1,
            args.Alpha,
            args.Lambda_2 if args.Lambda_2 > 0 else max(250 * 10 ** (-(epoch//1500)), 0.5),
            args.Beta,
            args.Loss_Type,
            args.has_mask,
            Criterion_L1 = torch.nn.L1Loss()
            )
        
        loss_dict["Train_gen_loss"].append(train_gen_loss)
        loss_dict["Train_disc_loss"].append(train_disc_loss)
        
        eval_gen_loss, eval_disc_loss, error, distance, psnr = eval_one_epoch(
            GEN, 
            DIS, 
            eval_dataloader, 
            args.device, 
            epoch,
            args.Lambda_1,
            args.Alpha,
            args.Lambda_2 if args.Lambda_2 > 0 else max(250 * 10 ** (-(epoch//1500)), 0.5),
            args.Beta,
            args.Loss_Type,
            args.has_mask,
            thres = False,
            Criterion_L1 = torch.nn.L1Loss()           
            )

        loss_dict["Eval_gen_loss"].append(eval_gen_loss)
        loss_dict["Eval_disc_loss"].append(eval_disc_loss)
        loss_dict["Eval_Error"].append(error)
        # loss_dict["Pearson_r_value"].append(pearson_r)
        loss_dict["Distance"].append(distance)
        loss_dict["PSNR"].append(psnr)
        
        if ((epoch + 1) % args.image_interval == 0) or (epoch == (args.num_epoches - 1)):
            save_example_image(GEN, eval_dataloader, args.device, epoch, image_save_route, model_detail)
            
        if ((epoch + 1) % args.checkpoint_interval == 0) or (epoch == (args.num_epoches - 1)):
            save_checkpoint(GEN, DIS, optimizer_G, optimizer_D, model_detail, epoch, checkpoint_folder)
        
        t1 = time.time()
        
        print("Time consumption for Epoch {} is {}:{:0>2d}:{:0>2d}\n".format(epoch, int(t1-t0)//3600, (int(t1-t0)%3600)//60, int(t1-t0)%60))
        
    save_model(GEN, DIS, loss_dict, model_detail, Folder_route, args.num_epoches)

    test_dataset = ImageDataset(path = args.test_path, output_type = args.out_slice, num_slice = args.num_slice, has_mask = args.has_mask, img_size = 256, scale_range = None, version = "test")
    
    test_dataloader = DataLoader(test_dataset, batch_size = 4, shuffle = False, num_workers = 4 if args.device == 'cpu' else 0)
    
    print("\n"+"*"*20)
    print("Performance on Test dataset for {} is ...".format(model_detail))
    test_model_performance(GEN, test_dataloader, Folder_route, model_detail, args.device)

    end_time = time.time()    

    print("\nTime consumption from Epoch {} to Epoch {} is {}:{:0>2d}:{:0>2d}".format(start_epoch+1, args.num_epoches, int(end_time - start_time)//3600, (int(end_time - start_time)%3600)//60, int(end_time - start_time)%60))
    
if __name__ == "__main__":
    args = parse_args()
    main(args)
    