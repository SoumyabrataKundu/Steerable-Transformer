#!/bin/bash


#SBATCH --job-name=RUNMODEDATASETrRADIUSkTHETA
#SBATCH --output=output.test
#SBATCH --error=error.test
#SBATCH --account=pi-risi
#SBATCH --partition=gm4
#SBATCH --nodes=1
#SBATCH --gres=gpu:GPU


module load python
source activate /home/soumyabratakundu/.conda/envs/conda_env


python view.py --model_path="./" --data_path="DATAPATH" --loss_type="LOSS" --batch_size=BATCHSIZE --n_radius=RADIUS --max_m=THETA
