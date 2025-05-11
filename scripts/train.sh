#!/bin/bash


#SBATCH --job-name=RUNDATASETrRADIUSkTHETA
#SBATCH --output=output.train
#SBATCH --error=error.train
#SBATCH --account=pi-risi
#SBATCH --partition=gm4
#SBATCH --nodes=1
#SBATCH --gres=gpu:GPU


module load python
source activate /home/soumyabratakundu/.conda/envs/conda_env

start_time=`date +%s`

python train.py --data_path="DATAPATH" --batch_size=BATCHSIZE --n_radius=RADIUS --max_m=THETA --learning_rate=0.005 --weight_decay=0.0005 --num_epochs=EPOCHS --lr_decay_rate=0.5 --lr_decay_schedule=20 --metric_type=METRICTYPE --save=SAVE

end_time=`date +%s`
runtime=$((end_time - start_time))
echo
echo Runtime = $((runtime / 3600))h $(((runtime % 3600) / 60))m
