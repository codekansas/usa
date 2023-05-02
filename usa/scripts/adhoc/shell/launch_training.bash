#!/bin/bash

# Where to log the launch information
results_dir=out
results_path=$results_dir/results.txt
echo "Results: $results_path"
mkdir -p $results_dir
rm -f $results_path

# How many hidden layers to use
num_layers=(
    # 2
    4
    # 8
)

# How many hidden dimensions to use
num_dims=(
    # 128
    # 256
    512
)

# Which configurations to run
clip_models=(
    # RN50
    ViT_B_16
)

# Which datasets to run
datasets=(
    lab_stretch
    kitchen_stretch
    chess_stretch
    replica_apt_3_mnp
    studio_r3d
    lab_r3d
    chris_lab
)

yaml_file=configs/clip_sdf/base.yaml

for num_layers_i in ${num_layers[@]}; do
    for num_dims_i in ${num_dims[@]}; do
        for clip_model in ${clip_models[@]}; do
            for dataset in ${datasets[@]}; do
                echo "====" | tee -a $results_path
                echo "Running $yaml_file" | tee -a $results_path
                echo " ↪ num_layers: $num_layers_i" | tee -a $results_path
                echo " ↪ num_dims: $num_dims_i" | tee -a $results_path
                echo " ↪ clip_model: $clip_model" | tee -a $results_path
                echo " ↪ dataset: $dataset" | tee -a $results_path
                echo "====" | tee -a $results_path

                python -m ml.scripts.cli mp_train \
                    $yaml_file \
                    model.num_layers=${num_layers_i} \
                    model.hidden_dims=${num_dims_i} \
                    task.dataset=${dataset} \
                    task.clip_model=${clip_model} \
                    task.finished.max_steps=100000 | tee -a $results_path

                # Avoid overburdening the scheduler
                sleep 5
            done
        done
    done
done
