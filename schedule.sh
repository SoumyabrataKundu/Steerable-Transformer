#!/bin/bash

# Hyperparameters
runs=(1 2)
datasets=("RMNIST")
n_radius=(2)
n_theta=(8 12)
interpolations=(1)
restricted=(0 1)
restore=0

# Job Parameters
script="train"
main_directory=$PWD
jobs_done=4
gpu=1
MAX_CONCURRENT_JOBS=28
data_path="/project2/risi/soumyabratakundu/Data"

wait_for_jobs() {
    while [ $(squeue -u $USER | tail -n +2 | wc -l) -ge $MAX_CONCURRENT_JOBS ]; do
        sleep 60
    done
}

# Training Parameters
epochs=150
get_batch_size() {
    local k=$1
    local res=$2
    #if [ "$k" -eq 16 ]; then
    #    if [ "$res" -eq 0 ]; then
    #        echo 60
    #    elif [ "$res" -eq 1 ]; then
    #        echo 80
    #    fi
    #    else
            echo 100
    #fi
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

    for order in "${interpolations[@]}"
    do
        for sim in "${runs[@]}"
        do
            for restrict in "${restricted[@]}"
            do
                run=$((order*10 + 2*sim - 1 + restrict + 10))
                mkdir run${run} 2>/dev/null
                cd run${run}

                for radius in "${n_radius[@]}"
                do
    
                    for theta in "${n_theta[@]}"
                    do
                        ((job_counter++))
                        if [ "$job_counter" -le "$jobs_done" ]; then
                            echo job ${job_counter} run${run} ${data} r${radius}k${theta} already submitted.
                            continue
                        fi
                        mkdir r${radius}k${theta} 2>/dev/null

                        cd r${radius}k${theta}

                        if [ ${script} == "train" ]; then
                            mkdir Steerable 2>/dev/null
                            cp -r ${main_directory}/../../Steerable/Steerable/nn/ Steerable/
                            mkdir Steerable/datasets/ 2>/dev/null
                            cp ${main_directory}/../../Steerable/Steerable/datasets/hdf5.py Steerable/datasets/

                            if [ ${restore} -eq 0 ]; then
                                cp ${main_directory}/datasets/${data}/model.py ./
                            fi
                        fi

                        if [ ${script} == "eval" ]; then
                            cp -r ${main_directory}/datasets/${data}/evaluator/ ./ 2>/dev/null
                            cp -r ${main_directory}/datasets/${data}/eval.sh ./ 
                            cp -r ${main_directory}/datasets/${data}/eval.py ./ 
                        fi

                        cp ${main_directory}/scripts/${script}.sh ./ 2>/dev/null
                        cp ${main_directory}/scripts/${script}.py ./ 2>/dev/null
    
                        sed -i "s/GPU/${gpu}/g" ${script}.sh
                        sed -i "s/RUN/${run}/g" ${script}.sh
                        sed -i "s/DATASET/${data:0:1}/g" ${script}.sh
                        sed -i "s/RADIUS/${radius}/g" ${script}.sh
                        sed -i "s/THETA/${theta}/g" ${script}.sh
                        sed -i "s/ORDER/${order}/g" ${script}.sh
                        sed -i "s/RESTRICTED/${restrict}/g" ${script}.sh
                        sed -i "s#DATAPATH#${data_path}/${data}/data#g" ${script}.sh
                        sed -i "s/BATCHSIZE/$(get_batch_size "$theta" "$restrict")/g" ${script}.sh
                        sed -i "s/EPOCHS/${epochs}/g" ${script}.sh
                        sed -i "s/RESTORE/${restore}/g" ${script}.sh

                        wait_for_jobs
 
                        echo job ${job_counter} ${script}-run${run} ${data} r${radius}k${theta}
                        sbatch ${script}.sh

                        cd ../
                    done
                done
                cd ../
            done
        done
    done
    cd ../
done

echo "All jobs Submitted!"

