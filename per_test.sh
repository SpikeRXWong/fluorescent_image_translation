#!/usr/bin/env bash
#SBATCH --partition=veryshort
#SBATCH --time=0-00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
##SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --job-name=PerTest
#SBATCH --mem=64G
#SBATCH -e Performance_test_err_%j.txt
#SBATCH -o Performance_test_out_%j.txt
#SBATCH --mail-type=end
#SBATCH --mail-user=ruixiong.wang@bristol.ac.uk

##module load libs/cudnn/10.1-cuda-10.0

## 10020142 cacgan_2
## 01281651 sacagan_1

time python performance_evaluation.py -sn 10020142 01281651 --device cpu -bs 8

printf "\n\n"
echo "Ended on: $(date)"