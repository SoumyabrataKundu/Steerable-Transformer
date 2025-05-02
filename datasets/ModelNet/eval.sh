#!/bin/bash


#SBATCH --job-name=eval
#SBATCH --output=output.eval
#SBATCH --error=error.eval
#SBATCH --account=pi-risi
#SBATCH --partition=gm4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1


module load python
source activate /home/soumyabratakundu/.conda/envs/conda_env

python eval.py --model_path="./"  --model_path="./" --data_path="DATAPATH" --batch_size=BATCHSIZE --n_radius=RADIUS --max_m=THETA --interpolation=ORDER --restricted=RESTRICTED

