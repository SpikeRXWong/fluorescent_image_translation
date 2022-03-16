# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 09:49:46 2021

@author: rw17789
"""

import torch
import torch.nn as nn
from torchvision.utils import save_image
import os

from utils import weight_def, Weighted_L1Loss, error_map_func, compute_ssim, compute_pearson, batch_intensity_compare, mean_calculater, median_calculater, compute_psnr

def train_one_epoch(GEN, DIS, dataloader, device, epoch, optimizer_G, optimizer_D,
                    DS_scale=5, Lambda1 = 0.2, Alpha = 0.8, Lambda2 = 0.2, Beta = 10,
                    Criterion_type = "DCGAN", has_mask = False, Criterion=None, Criterion_L1=None, Lambda_gp=10):
    
    assert Criterion_type in ["DCGAN", "WGAN", "Hinge"]
    if Criterion == None:
        if Criterion_type == "DCGAN":
            Criterion = nn.BCEWithLogitsLoss()
        elif Criterion_type == "Hinge":
            Criterion = nn.ReLU()

    if Criterion_L1 == None:
        Criterion_L1 = Weighted_L1Loss
        
    if has_mask:
        Mask_Criterion = nn.BCEWithLogitsLoss() if Criterion_type == "DCGAN" else nn.ReLU()
          
    GEN.train()
    for key in DIS.keys():
        DIS[key].train()
    
    disc_loss = {"Image": 0}
    if has_mask:
        disc_loss["Mask"] = 0
    nd = 0
    gen_loss ={"Total_loss": 0, "G_gent_loss": 0, "L1_loss":0, "SSIM_loss":0}
    if has_mask:
        gen_loss["mask_loss"] = 0
        gen_loss["G_gent_mask_loss"] = 0
    ng = 0
        
    for idx, data in enumerate(dataloader):
                  
        x = data["brightfield"].to(device)
        y = data["fluorescent"].to(device)
        if has_mask:
            mask = data["mask"].to(device)
            weight = data["weight"].to(device)
        
        out = GEN(x)
        y_gent = out["image"].to(device)
        if has_mask:
            mask_gent = out["mask"].to(device)
     
        y_rand = torch.cat([y[1:,...],y[:1,...]],0) if y.size(0) !=  1 else y.permute(0,1,3,2)
        
        for _ in range(DS_scale):
        
            D_real = DIS["Image"](x, y)
            D_gent = DIS["Image"](x, y_gent.detach())
            D_rand = DIS["Image"](x, y_rand)
            
            rand_w = weight_def(epoch)
            
            if Criterion_type == "DCGAN":
                D_real_loss = Criterion(D_real, torch.ones_like(D_real))
                D_gent_loss = Criterion(D_gent, torch.zeros_like(D_gent))
                D_rand_loss = Criterion(D_rand, torch.zeros_like(D_rand))
                
                D_loss = (D_real_loss + D_gent_loss + D_rand_loss*rand_w) / (2 + rand_w)
            elif Criterion_type == "WGAN":
                bs, C, H, W = y.shape
                alpha = torch.rand(bs, 1, 1, 1).repeat(1, C, H, W).to(device)
                y_fake = (y_gent.detach() + y_rand * rand_w)/(1 + rand_w)
                y_mixed = y * alpha + y_fake * (1 - alpha)
                y_mixed.requires_grad_(True)
                mixed_scores = DIS["Image"](x, y_mixed)
                gradient = torch.autograd.grad(
                    inputs = y_mixed,
                    outputs = mixed_scores,
                    grad_outputs = torch.ones_like(mixed_scores),
                    create_graph = True,
                    retain_graph = True,
                    )[0]
                assert gradient.requires_grad
                gradient = gradient.view(gradient.shape[0], -1)
                gradient_norm = gradient.norm(2, dim=1)
                gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
                D_fake = (D_gent + D_rand * rand_w) / (1 + rand_w)
                
                D_loss = (
                    - (D_real.mean() - D_fake.mean()) + Lambda_gp * gradient_penalty
                    )
            else:
                D_fake = (D_gent + D_rand * rand_w) / (1 + rand_w)
                D_loss = Criterion(1.0 - D_real).mean() + Criterion(1.0 + D_fake).mean()
            
            optimizer_D["Image"].zero_grad()
            D_loss.backward()
            optimizer_D["Image"].step()
            
            if has_mask:
                D_mask_real = DIS["Mask"](mask.float())
                D_mask_gent = DIS["Mask"](mean_calculater(mask_gent,dim=1,soft_max=True,resize=False).detach())
                
                if Criterion_type == "DCGAN": 
                    D_mask_real_loss = Mask_Criterion(D_mask_real, torch.ones_like(D_mask_real))
                    D_mask_gent_loss = Mask_Criterion(D_mask_gent, torch.zeros_like(D_mask_gent))
                    D_mask_loss = D_mask_real_loss + D_mask_gent_loss
                else:
                    D_mask_loss = Mask_Criterion(1.0 - D_mask_real).mean() + Mask_Criterion(1.0 + D_mask_gent).mean()
                
                optimizer_D["Mask"].zero_grad()
                D_mask_loss.backward()
                optimizer_D["Mask"].step()
                
                disc_loss["Mask"] += D_mask_loss.item()
            
            disc_loss["Image"] += D_loss.item()
            nd += 1
        
        D_gent = DIS["Image"](x, y_gent)
        if Criterion_type == "DCGAN":
            G_gent_loss = Criterion(D_gent, torch.ones_like(D_gent))
        else:
            G_gent_loss = - D_gent.mean()
            
        if has_mask:
            D_mask_gent = DIS["Mask"](mean_calculater(mask_gent,dim=1,soft_max=True,resize=False))
            G_gent_mask_loss = Mask_Criterion(D_mask_gent, torch.ones_like(D_mask_gent)) if Criterion_type == "DCGAN" else - D_mask_gent.mean()

        L1_loss = Criterion_L1(y_gent, y)
        SSIM_loss = 1 - compute_ssim(y_gent, y)
        
        G_loss = G_gent_loss + ((1 - Alpha) * L1_loss + Alpha * SSIM_loss) * Lambda1
        
        if has_mask:
            mask_loss = torch.sum(nn.CrossEntropyLoss(weight = torch.tensor([1,3,9]).float().to(device), reduction = "none")(mask_gent, mask) * weight) / weight.sum()
            G_loss = G_loss + (mask_loss + G_gent_mask_loss * Beta) * Lambda2
        
        optimizer_G.zero_grad()
        G_loss.backward()
        optimizer_G.step()
        
        gen_loss["Total_loss"] += G_loss.item()
        gen_loss["G_gent_loss"] += G_gent_loss.item()
        gen_loss["L1_loss"] += L1_loss.item()
        gen_loss["SSIM_loss"] += SSIM_loss.item()
        if has_mask:
            gen_loss["mask_loss"] += mask_loss.item()
            gen_loss["G_gent_mask_loss"] += G_gent_mask_loss.item()

        del x, out, y, y_gent, y_rand, D_real, D_rand, D_gent
        if has_mask:
            del mask, weight, mask_gent, D_mask_gent, D_mask_real

        ng += 1
        
    assert nd == (ng * DS_scale)
    
    for key in gen_loss.keys():
        gen_loss[key] = gen_loss[key]/ng
        print("Generator loss {} (Epoch {}) : {:.4f} ".format(key, epoch, gen_loss[key]))
    for key in disc_loss.keys():
        disc_loss[key] = disc_loss[key]/nd
        print("Discriminator loss {} (Epoch {}) : {:.4f}".format(key, epoch, disc_loss[key]))
    
    return gen_loss, disc_loss
    

def eval_one_epoch(GEN, DIS, dataloader, device, epoch, Lambda1 = 0.2, Alpha = 0.8, Lambda2 = 0.2, Beta = 10,
                   Criterion_type = "DCGAN", has_mask = False, thres = False, Criterion=None, Criterion_L1=None):
    assert Criterion_type in ["DCGAN", "WGAN", "Hinge"]
    if Criterion == None:
        if Criterion_type == "DCGAN":
            Criterion = nn.BCEWithLogitsLoss()
        elif Criterion_type == "Hinge":
            Criterion = nn.ReLU()
    if Criterion_L1 == None:
        Criterion_L1 = Weighted_L1Loss
        
    if has_mask:
        Mask_Criterion = nn.BCEWithLogitsLoss() if Criterion_type == "DCGAN" else nn.ReLU()          
          
    GEN.eval()
    for key in DIS.keys():
        DIS[key].eval()
    
    disc_loss = {"Image": 0}
    if has_mask:
        disc_loss["Mask"] = 0
    gen_loss ={"Total_loss": 0, "G_gent_loss": 0, "L1_loss":0, "SSIM_loss":0}
    if has_mask:
        gen_loss["mask_loss"] = 0
        gen_loss["G_gent_mask_loss"] = 0
    error = 0
    # pearson_r = 0
    distance = 0
    psnr = 0
    n = 0
    
    with torch.no_grad():        
        for idx, data in enumerate(dataloader):
            
            x = data["brightfield"].to(device)
            y = data["fluorescent"].to(device)
            if has_mask:
                mask = data["mask"].to(device)
                weight = data["weight"].to(device)
            
            out = GEN(x)
            y_gent = out["image"].to(device)
            if has_mask:
                mask_gent = out["mask"].to(device)
            
            y_rand = torch.cat([y[1:,...],y[:1,...]],0) if y.size(0) !=  1 else y.permute(0,1,3,2)
            
            D_real = DIS["Image"](x, y)
            D_gent = DIS["Image"](x, y_gent)
            D_rand = DIS["Image"](x, y_rand)
            
            rand_w = weight_def(epoch)
            
            if Criterion_type == "DCGAN":
                D_real_loss = Criterion(D_real, torch.ones_like(D_real))
                D_gent_loss = Criterion(D_gent, torch.zeros_like(D_gent))
                D_rand_loss = Criterion(D_rand, torch.zeros_like(D_rand))
                
                D_loss = (D_real_loss + D_gent_loss + D_rand_loss*rand_w) / (2 + rand_w)
            elif Criterion_type == "WGAN":
                D_fake = (D_gent + D_rand * rand_w) / (1 + rand_w)
                
                D_loss = - (D_real.mean() - D_fake.mean())
            else:
                D_fake = (D_gent + D_rand * rand_w) / (1 + rand_w)
                D_loss = Criterion(1.0 - D_real).mean() + Criterion(1.0 + D_fake).mean()
                            
            disc_loss["Image"] += D_loss.item()
            
            if has_mask:
                D_mask_real = DIS["Mask"](mask.float())
                D_mask_gent = DIS["Mask"](median_calculater(mask_gent,dim=1,soft_max=True,resize=False).float())
                
                if Criterion_type == "DCGAN": 
                    D_mask_real_loss = Mask_Criterion(D_mask_real, torch.ones_like(D_mask_real))
                    D_mask_gent_loss = Mask_Criterion(D_mask_gent, torch.zeros_like(D_mask_gent))
                    D_mask_loss = D_mask_real_loss + D_mask_gent_loss
                else:
                    D_mask_loss = Mask_Criterion(1.0 - D_mask_real).mean() + Mask_Criterion(1.0 + D_mask_gent).mean()
                
                disc_loss["Mask"] += D_mask_loss.item()
            
            if Criterion_type == "DCGAN":
                G_gent_loss = Criterion(D_gent, torch.ones_like(D_gent))
            else:
                G_gent_loss = - D_gent.mean()
            
            if has_mask:
                G_gent_mask_loss = Mask_Criterion(D_mask_gent, torch.ones_like(D_mask_gent)) if Criterion_type == "DCGAN" else - D_mask_gent.mean()
                
            L1_loss = Criterion_L1(y_gent, y)
            SSIM_loss = 1 - compute_ssim(y_gent, y)
            G_loss = G_gent_loss + ((1 - Alpha) * L1_loss + Alpha * SSIM_loss) * Lambda1
            
            if has_mask:
                mask_loss = torch.sum(nn.CrossEntropyLoss(weight = torch.tensor([1,3,9]).float().to(device), reduction = "none")(mask_gent, mask) * weight) / weight.sum()
                G_loss = G_loss + (mask_loss + G_gent_mask_loss * Beta) * Lambda2
            
            gen_loss["Total_loss"] += G_loss.item()
            gen_loss["G_gent_loss"] += G_gent_loss.item()
            gen_loss["L1_loss"] += L1_loss.item()
            gen_loss["SSIM_loss"] += SSIM_loss.item()
            if has_mask:
                gen_loss["mask_loss"] += mask_loss.item()
                gen_loss["G_gent_mask_loss"] += G_gent_mask_loss.item()
            
            error += error_map_func(y_gent, y, mode="error")
            
            # pearson_r += compute_pearson(y_gent, y, threshold = thres)
            
            distance += SSIM_loss

            psnr += compute_psnr(y_gent.sum(dim=1, keepdim=True), y.sum(dim=1, keepdim=True), 2)

            del x, out, y, y_gent, y_rand, D_real, D_gent, D_rand
            if has_mask:
                del mask, weight, mask_gent, D_mask_gent, D_mask_real
            
            n += 1
    
    for key in gen_loss.keys():
        gen_loss[key] = gen_loss[key]/n
        print("Generator loss {} (Epoch {}) : {:.4f} ".format(key, epoch, gen_loss[key]))

    for key in disc_loss.keys():
        disc_loss[key] = disc_loss[key]/n
        print("Discriminator evaluation loss {} (Epoch {}) : {:.4f}".format(key, epoch, disc_loss[key]))

    error = error/n
    # pearson_r = pearson_r/n
    distance = distance/n
    psnr = psnr/n
      
    print("Error from Ground Truth (Epoch {}) : {:.4f}".format(epoch, error))
    # print("Pearson r value between generated image and ground ture (Epoch {}) : {:.4f}".format(epoch, pearson_r))
    print("Distance from the Ground truth: {:.4f}".format(distance))
    print("PSNR value for groud truth and generated image is {:.4f}".format(psnr))
    
    return gen_loss, disc_loss, error, distance, psnr

def save_model(model_G, model_D, loss_dict, filename, destination, finish_epoch):
    
    model_G = model_G.to("cpu")
    for key in model_D.keys():
        model_D[key] = model_D[key].to("cpu")
    
    model = {
        "Model_detail": filename,
        "Generator": model_G.state_dict(),
        "Discriminator": {"Image": model_D["Image"].state_dict()},
        "Loss_Dict": loss_dict
        }
    if "Mask" in model_D.keys():
        model["Discriminator"]["Mask"] = model_D["Mask"].state_dict()
    
    if not os.path.isdir(destination):
        os.mkdir(destination)
        
    save_path = os.path.join(destination, "{}_E{:0>4d}.pth.tar".format(filename, finish_epoch))
    torch.save(model, save_path)
    
    print("Model saved!")
    

def save_checkpoint(model_G, model_D, optimizer_G, optimizer_D, filename, epoch, destination):
    
    print("Chechpoint for Generator and Discriminator of {} at Epoch {} saved!".format(filename, epoch))
    
    checkpoint = {
        "Model_detail": filename,
        "Epoch": epoch,
        
        "Generator_model": model_G.state_dict(),
        "Generator_optimizer": optimizer_G.state_dict(),
        # "Generator_lr_schedular": lr_scheduler_G.state_dict() if lr_scheduler_G is not None else None,
        
        "Discriminator_model": {"Image": model_D["Image"].state_dict()},
        "Discriminator_optimizer": {"Image": optimizer_D["Image"].state_dict()},        
        # "Discriminator_lr_schedular": lr_scheduler_D.state_dict() if lr_scheduler_D is not None else None
        
        }
    if "Mask" in model_D.keys():
        checkpoint["Discriminator_model"]["Mask"] = model_D["Mask"].state_dict()
        checkpoint["Discriminator_optimizer"]["Mask"] = optimizer_D["Mask"].state_dict()
    
    if not os.path.isdir(destination):
        os.mkdir(destination)
        
    checkpoint_path = os.path.join(destination, "Checkpoint_{}_E{:0>4d}.pth.tar".format(filename, epoch))
    torch.save(checkpoint, checkpoint_path)
    
def optimizer2device(x, device):
    for state in x.state.values():
        for k,v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    return x

def save_example_image(model, dataloader, device, epoch, save_route, filename):
    data = next(iter(dataloader))
    x = data["brightfield"]
    y = data["fluorescent"]
    
    x = x.to(device)
    y = y.to(device)
    c = x.shape[1]//2
    
    if not os.path.isdir(save_route):
        os.mkdir(save_route)
    savename = os.path.join(save_route, "Example_image_{}_E{:0>4d}.png".format(filename, epoch))
    
    model.eval()
    with torch.no_grad():
        out = model(x)
        y_gent = out["image"]
        if "mask" in out.keys():
            mask_gent = out["mask"]
        
    y_show = y.max(dim=1,keepdim=True)[0].repeat(1,3,1,1)
    if y.shape[1] == 2:
        y_gent_show = torch.cat([y_gent, -torch.ones_like(y_gent[:,:1,:,:])], dim=1)
    else:
        y_gent_show = y_gent.max(dim=1, keepdim=True)[0].repeat(1,3,1,1)
    
    error_map = error_map_func(y_gent, y)
    # pearson_r_map = batch_intensity_compare(y_gent.max(dim=1,keepdim=True)[0], y.max(dim=1,keepdim=True)[0])
    
    # image = torch.cat([x[:, c:c+1, ...].repeat(1,3,1,1), y_show, y_gent_show, error_map, pearson_r_map], di!m = 0)
    image = torch.cat([x[:, c:c+1, ...].repeat(1,3,1,1), y_show, y_gent_show, error_map], dim = 0)

    if "mask" in out.keys():
        mask_gt = (data["mask"].repeat(1,3,1,1) - 1).to(device)
        mask_show = median_calculater(mask_gent, dim=1, soft_max = True, resize = True, value_range = (-1,1)).repeat(1,3,1,1)
        image = torch.cat([image, mask_gt, mask_show], dim = 0)
    
    image.to("cpu")
    
    try:
        save_image(image, savename, nrow=4, padding = 10, normalize=True, value_range=(-1,1), pad_value = 1)
    except TypeError:
        save_image(image, savename, nrow=4, padding = 10, normalize=True, range=(-1,1), pad_value = 1)

def test_model_performance(model_G, dataloader, folder, model_detail, device):
    test_items = ["L1_Loss", "MSE_Loss", "PSNR", "SSIM"]
    Test = {}
    model_G = model_G.to(device)
    model_G.eval()
    for key in test_items:
        Test[key] = []
    for idx, data in enumerate(dataloader, 1):
        x = data["brightfield"].to(device)
        y_gt = data["fluorescent"].to(device)
        with torch.no_grad():
            out = model_G(x)
        y_gent = out["image"]

        Test["L1_Loss"].append(torch.nn.functional.l1_loss(y_gt, y_gent, reduction = 'none').mean(dim = (1,2,3)))
        Test["MSE_Loss"].append(torch.nn.functional.mse_loss(y_gt, y_gent, reduction = 'none').mean(dim = (1,2,3)))
        Test["PSNR"].append(compute_psnr(y_gt, y_gent, 2, "none"))
        Test["SSIM"].append(compute_ssim(y_gt, y_gent, size_average = False))

        if "mask" in out.keys():
            mask_gent = out["mask"]
            
        c = x.shape[1]//2
        if y_gt.shape[1] == 2:
            y_show = torch.cat([y_gt, y_gt[:,:1, ...]], dim=1)
            y_gent_show = torch.cat([y_gent, y_gent[:,:1, ...]], dim=1)
        else:
            y_show = y_gt.repeat(1,3,1,1)
            y_gent_show = y_gent.repeat(1,3,1,1)  
        
        image = torch.cat([x[:, c:c+1, ...].repeat(1,3,1,1), y_show, y_gent_show], dim = 0)
        
        if "mask" in out.keys():
            mask_show = median_calculater(mask_gent, dim=1, soft_max = True, resize = True, value_range = (-1,1)).repeat(1,3,1,1)
            mask_gt = data["mask"].repeat(1,3,1,1) -1
            image = torch.cat([image, mask_gt, mask_show], dim = 0)
        
        for idxx in range(len(y_gt)):
            savename = os.path.join(folder, "brightfield_{}_{}_{}_error.png".format(model_detail, idx, idxx+1))
            save_image(x[idxx, c, ...], savename, nrow=1, padding=0, normalize=True, value_range=(-1,1))
            
            savename = os.path.join(folder, "fluorescent_gt_{}_{}_{}.png".format(model_detail, idx, idxx + 1))
            save_image(y_gt[idxx].max(dim=0)[0], savename, nrow=1, padding=0, normalize=True, value_range=(-1,1))
                
            savename = os.path.join(folder, "fluorescent_tl_{}_{}_{}.png".format(model_detail, idx, idxx + 1))
            save_image(y_gent[idxx].max(dim=0)[0], savename, nrow=1, padding=0, normalize=True, value_range=(-1,1))
                
            if y_gt.shape[1] == 2:
                savename = os.path.join(folder, "fluorescent_gt_{}_{}_{}_color.png".format(model_detail, idx, idxx + 1))
                save_image(y_show[idxx], savename, nrow=1, padding=0, normalize=True, value_range=(-1,1))
            
                savename = os.path.join(folder, "fluorescent_tl_{}_{}_{}_color.png".format(model_detail, idx, idxx + 1))
                save_image(y_gent_show[idxx], savename, nrow=1, padding=0, normalize=True, value_range=(-1,1))
            
            if "mask" in out.keys():
                savename = os.path.join(folder, "mask_gt_{}_{}_{}.png".format(model_detail, idx, idxx + 1))
                save_image(mask_gt[idxx, 0, ...].float(), savename, nrow=1, padding=0, normalize=True, value_range=(-1,1))
                
                savename = os.path.join(folder, "mask_tl_{}_{}_{}.png".format(model_detail, idx, idxx + 1))
                save_image(mask_show[idxx, 0, ...].float(), savename, nrow=1, padding=0, normalize=True, value_range=(-1,1)) 
        
        savename = os.path.join(folder, "Test_image_{}_{}.png".format(model_detail, idx))
        save_image(image, savename, nrow=len(y_gt), padding = 10, normalize=True, value_range=(-1,1), pad_value = 1)
        
        error_map = error_map_func(y_gent, y_gt)
        
        image = torch.cat([image, error_map], dim = 0)
        
        # image = image.to("cpu")
        savename = os.path.join(folder, "Test_image_{}_{}_error.png".format(model_detail, idx))
        save_image(image, savename, nrow=4, padding = 10, normalize=True, value_range=(-1,1), pad_value = 1)
        
        if "mask" in out.keys():
            del mask_gent, mask_gt, mask_show
        del x, y_gt, y_gent, y_show, y_gent_show, out

    for key in Test:
        Test[key] = torch.cat(Test[key], dim=0)
        print("{} : {:.4f} ({:.4f} - {:.4f})".format(key, Test[key].mean(), Test[key].min(), Test[key].max()))
    torch.save(Test, os.path.join(folder, "Test_result_{}.pth".format(model_detail)))     