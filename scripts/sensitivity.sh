#!/bin/bash


#SBATCH --job-name=RUNDATASETrRADIUSkTHETA
#SBATCH --output=output.test
#SBATCH --error=error.test
#SBATCH --account=pi-risi
#SBATCH --partition=gm4
#SBATCH --nodes=1
#SBATCH --gres=gpu:GPU


module load python
source activate /home/soumyabratakundu/.conda/envs/conda_env

start_time=`date +%s`

python sensitivity.py --model_path="./" --data_path="DATAPATH" --batch_size=BATCHSIZE --n_radius=RADIUS --max_m=THETA --interpolation=ORDER --restricted=RESTRICTED

end_time=`date +%s`
runtime=$((end_time - start_time))
echo Runtime = $((runtime / 3600))h $(((runtime % 3600) / 60))m
