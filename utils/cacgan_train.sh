#!/usr/bin/env bash
#SBATCH --partition=gpu###_veryshort
#SBATCH --time=1-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=CACGAN
#SBATCH --mem=16G
#SBATCH -e CACGAN_gpu_err_%j.txt
#SBATCH -o CACGAN_gpu_out_%j.txt
#SBATCH --mail-type=end
#SBATCH --mail-user=ruixiong.wang@bristol.ac.uk

module load libs/cudnn/10.1-cuda-10.0

time python cacgan_train.py -n 4500 --GAP 11110 --DAL two --out_slice 1 -s 13 --down_step 5 --Loss_Type DCGAN --image_interval 50 --checkpoint_interval 250 \
-lg 1e-5 -ld 1e-4 --Lambda_1 50 --Alpha 0.8 --Lambda_2 2.5 --Beta 0.2 \

printf "\n\n"
echo "Ended on: $(date)"