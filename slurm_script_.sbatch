#!/bin/bash
#SBATCH --job-name=chexpert                     # Job name
#SBATCH -N1 --gres=gpu:H200:1                   # Number of nodes and cores per node required
#SBATCH --mem-per-gpu=224GB                     # Memory per CPU core, 8CPUs/GPU
#SBATCH -t16:00:00                              # Duration of the job
#SBATCH -oReport-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=zzhai37@gatech.edu          # E-mail address for notifications
# Load required modules explicitly
# source /home/hice1/zzhai37/scratch/.bashrc
# source activate chexpert
source activate nsp
# Move to working directory
cd /home/hice1/zzhai37/scratch/nsp/cheXpert
# Execute Python script within "chexpert" environment
python cnn_.py