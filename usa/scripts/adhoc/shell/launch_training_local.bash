#!/bin/bash

# Where to log the launch information
results_dir=out
results_path=$results_dir/results.txt
echo "Results: $results_path"
mkdir -p $results_dir
rm -f $results_path

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

for dataset in ${datasets[@]}; do
    echo "====" | tee -a $results_path
    echo "Running $yaml_file on $dataset" | tee -a $results_path
    echo "====" | tee -a $results_path

    python -m ml.scripts.cli train \
        configs/clip_sdf/base.yaml \
        model.num_layers=4 \
        model.hidden_dims=512 \
        task.dataset=${dataset} \
        task.clip_model=ViT_B_16 \
        task.finished.max_steps=100000 | tee -a $results_path

    # Avoid overburdening the scheduler
    sleep 5
done
