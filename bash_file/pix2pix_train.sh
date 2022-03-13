#!/usr/bin/env bash
#SBATCH --partition=veryshort
#SBATCH --time=0-00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=Pix2Pix
#SBATCH --mem=16G
#SBATCH -e Pix2Pix_gpu_err_%j.txt
#SBATCH -o Pix2Pix_gpu_out_%j.txt
#SBATCH --mail-type=end
#SBATCH --mail-user=ruixiong.wang@bristol.ac.uk

module load libs/cudnn/10.1-cuda-10.0

time python pix2pix_compare.py -n 0 -s 13 --Loss_Type DCGAN --image_interval 50 --checkpoint_interval 250 \
-lg 1e-5 -ld 1e-4 --Lambda_1 50 --Alpha 0 \
--RESUME -c /mnt/storage/scratch/rw17789/CACGAN_Result/Pixel2Pixel_compare_model_10055184_02211735/Checkpoint_Storage_Pixel2Pixel_compare_model_10055184_02211735/Checkpoint_Pixel2Pixel_compare_model_10055184_02211735_E4499.pth.tar


printf "\n\n"
echo "Ended on: $(date)"