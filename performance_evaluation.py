import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from dbcelldataset import DBCImageDataset as ImageDataset
from xacgan_network import Generator
from utils import parameter_extractor, compute_psnr, compute_ssim, error_map_func, median_calculater

import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description = "Parameter setting")
    
    parser.add_argument("--model_path", type = str)
    
    parser.add_argument("--Destination", type=str)
    
    parser.add_argument("--test_path", type=str)
    
    parser.add_argument("--device", default="cuda", type=str, choices=["cpu", "cuda"])

    parser.add_argument("-bs", "--batch_size", default=4, type=int, choices=[4, 8])
    
    return parser.parse_args()

def main(args):        
    if not torch.cuda.is_available():
        args.device = "cpu"
        
    Test = {}
    test_items = ["L1_Loss", "MSE_Loss", "PSNR", "SSIM"]
    
    save_folder = os.path.join(args.Destination, "Performance_Test_{}".format(os.environ['SLURM_JOBID']))
    os.makedirs(save_folder)
    
    model_file = torch.load(args.model_path, map_location = "cpu")
    
    num_slice, out_slice, GAP, _, down_step, _ = parameter_extractor(model_file["Model_detail"])
    GEN = Generator(in_channels = num_slice, out_channels = out_slice, attn_pos=GAP, has_mask = model_name[0] != "S", down_step = down_step, bypass = False)
    GEN.load_state_dict(torch.load(args.model_state_dict))
    GEN = GEN.to(args.device)
    GEN.eval()

    del model_file

    test_dataset = ImageDataset(
        path = args.test_path, 
        output_type = out_slice, 
        num_slice = num_slice, 
        has_mask = model_name[0] == "C", 
        img_size = 256, 
        scale_range = None, 
        version = "test"
        )

    test_dataloader = DataLoader(
        test_dataset, 
        batch_size = args.batch_size, 
        shuffle = False, 
        num_workers = args.batch_size if args.device == 'cpu' else 0
        )

    for key in test_items:
        Test[key] = []

    for idx, data in enumerate(test_dataloader, 1):
        x = data["brightfield"].to(args.device)
        y_gt = data["fluorescent"].to(args.device)
        with torch.no_grad():
            out = GEN(x)
        y_gent = out["image"]

        Test["L1_Loss"].append(torch.nn.functional.l1_loss(y_gt, y_gent, reduction = 'none').mean(dim = (1,2,3)))
        Test["MSE_Loss"].append(torch.nn.functional.mse_loss(y_gt, y_gent, reduction = 'none').mean(dim = (1,2,3)))
        Test["PSNR"].append(compute_psnr(y_gt, y_gent, 2, "none"))
        Test["SSIM"].append(compute_ssim(y_gt, y_gent, size_average = False))

        if "mask" in out.keys():
            mask_gent = out["mask"]

        c = x.shape[1]//2
        if y_gt.shape[1] == 2:
            y_show = torch.cat([y_gt, y_gt[:,:1, ...]], dim=1) #-torch.ones_like(y_gt[:,:1,:,:])],dim=1)
            y_gent_show = torch.cat([y_gent, y_gent[:,:1, ...]], dim=1)#-torch.ones_like(y_gent[:,:1,:,:])], dim=1)
        else:
            y_show = y_gt.repeat(1,3,1,1)
            y_gent_show = y_gent.repeat(1,3,1,1)

        image = torch.cat([x[:, c:c+1, ...].repeat(1,3,1,1), y_show, y_gent_show], dim = 0)

        if "mask" in out.keys():
            mask_show = median_calculater(mask_gent, dim=1, soft_max = True, resize = True, value_range = (-1,1)).repeat(1,3,1,1)
            mask_gt = data["mask"].repeat(1,3,1,1) -1
            image = torch.cat([image, mask_gt, mask_show], dim = 0)

        error_map = error_map_func(y_gent, y_gt)

        for idxx in range(len(y_gt)):
            savename = os.path.join(save_folder, "brightfield_{}_{}_{}.png".format(sn, idx, idxx + 1))
            save_image(x[idxx, c, ...], savename, nrow=1, padding=0, normalize=True, value_range=(-1,1))

            savename = os.path.join(save_folder, "fluorescent_gt_{}_{}_{}.png".format(sn, idx, idxx + 1))
            save_image(y_gt[idxx].max(dim=0)[0], savename, nrow=1, padding=0, normalize=True, value_range=(-1,1))

            savename = os.path.join(save_folder, "fluorescent_tl_{}_{}_{}.png".format(sn, idx, idxx + 1))
            save_image(y_gent[idxx].max(dim=0)[0], savename, nrow=1, padding=0, normalize=True, value_range=(-1,1))

            if y_gt.shape[1] == 2:
                savename = os.path.join(save_folder, "fluorescent_gt_{}_{}_{}_color.png".format(sn, idx, idxx + 1))
                save_image(y_show[idxx], savename, nrow=1, padding=0, normalize=True, value_range=(-1,1))

                savename = os.path.join(save_folder, "fluorescent_tl_{}_{}_{}_color.png".format(sn, idx, idxx + 1))
                save_image(y_gent_show[idxx], savename, nrow=1, padding=0, normalize=True, value_range=(-1,1))

            if "mask" in out.keys():
                savename = os.path.join(save_folder, "mask_gt_{}_{}_{}.png".format(sn, idx, idxx + 1))
                save_image(mask_gt[idxx, 0, ...].float(), savename, nrow=1, padding=0, normalize=True, value_range=(-1,1))

                savename = os.path.join(save_folder, "mask_tl_{}_{}_{}.png".format(sn, idx, idxx + 1))
                save_image(mask_show[idxx, 0, ...].float(), savename, nrow=1, padding=0, normalize=True, value_range=(-1,1))           

        # image = image.to("cpu")
        savename = os.path.join(save_folder, "Output_image_{}_{}_{}.png".format(sn, idx, idxx + 1))
        save_image(image, savename, nrow=args.batch_size, padding = 10, normalize=True, value_range=(-1,1), pad_value = 1)

        image = torch.cat([image, error_map], dim = 0)
        savename = os.path.join(save_folder, "Output_image_{}_{}_{}_error.png".format(sn, idx, idxx + 1))
        save_image(image, savename, nrow=args.batch_size, padding = 10, normalize=True, value_range=(-1,1), pad_value = 1)

        if "mask" in out.keys():
            del mask_gent, mask_gt, mask_show
        del x, y_gt, y_gent, y_show, y_gent_show, out, savename        

    for key in Test:
        Test[key] = torch.cat(Test[key], dim = 0)
        print("{} : {:.4f} ({:.4f} - {:.4f})".format(key, Test[key].mean(), Test[key].min(), Test[key].max()))
    print("\n")
        
    torch.save(Test, os.path.join(save_folder, "Test_Result_{}.pth".format(os.environ['SLURM_JOBID'])))
            

if __name__ == "__main__":
    args = parse_args()
    main(args)
