#!/bin/bash

# Experiment
runs=(1)
datasets=("RMNIST")
script="test"
metric_type="accuracy"
save=0

# Model Hyperparameters
radius=2
theta=8
batch_size=100
epochs=3

# Job Parameters
main_directory=$PWD
data_path="${PWD}/../Data"
jobs_done=0
gpu=1
MAX_CONCURRENT_JOBS=28

wait_for_jobs() {
    while [ $(squeue -u $USER | tail -n +2 | wc -l) -ge $MAX_CONCURRENT_JOBS ]; do
        sleep 60
    done
}

trap 'echo "Script interrupted. Jobs submitted so far: $((${job_counter}-1))"; exit' SIGINT

mkdir experiment_runs 2>/dev/null
cd experiment_runs

echo Jobs Submitted = ${jobs_done}
job_counter=0

for data in "${datasets[@]}"
do
    mkdir ${data} 2>/dev/null
    cd ${data}

    for sim in "${runs[@]}"
    do
        run=${sim}
        mkdir run${run} 2>/dev/null
        cd run${run}

        ((job_counter++))
        if [ "$job_counter" -le "$jobs_done" ]; then
            echo job ${job_counter} run${run} ${data} r${radius}k${theta} already submitted.
            continue
        fi

        ## Copy Files
        if [ ${script} == "train" ]; then
            cp -r ${main_directory}/../Steerable/Steerable/ ./
            cp ${main_directory}/datasets/${data}/model.py ./
        fi
        cp ${main_directory}/scripts/${script}.sh ./ 2>/dev/null
        cp ${main_directory}/scripts/${script}.py ./ 2>/dev/null
    
        ## Modify Script
        sed -i "s/GPU/${gpu}/g" ${script}.sh
        sed -i "s/RUN/${run}/g" ${script}.sh
        sed -i "s/DATASET/${data:0:1}/g" ${script}.sh
        sed -i "s/RADIUS/${radius}/g" ${script}.sh
        sed -i "s/THETA/${theta}/g" ${script}.sh
        sed -i "s/LOSS/${loss}/g" ${script}.sh
        sed -i "s#DATAPATH#${data_path}/${data}/data#g" ${script}.sh
        sed -i "s/BATCHSIZE/${batch_size}/g" ${script}.sh
        sed -i "s/EPOCHS/${epochs}/g" ${script}.sh
        sed -i "s/METRICTYPE/${metric_type}/g" ${script}.sh
        sed -i "s/SAVE/${save}/g" ${script}.sh

        wait_for_jobs

        echo job ${job_counter} ${script}-run${run} ${data}
        sbatch ${script}.sh

        cd ../
    done
    cd ../
done

echo "All jobs Submitted!"

