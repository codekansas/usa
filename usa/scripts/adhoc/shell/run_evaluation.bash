#!/bin/bash

# Kill script on first error.
set -e

# Ensures that $EVAL_RUN_DIR is set.
if [ -z "$EVAL_RUN_DIR" ]; then
    echo "EVAL_RUN_DIR is not set"
    exit 1
fi

# Toggles using Slurm.
use_slurm=1

slurm_partition=devlab
# slurm_partition=learnlab

eval_sets=(
    lab_stretch
    kitchen_stretch
    chess_stretch
    replica_apt_3_mnp
    studio_r3d
    lab_r3d
    chris_lab
)

ckpts_paths=$EVAL_RUN_DIR/language-navigation-ckpts
zip_url=https://github.com/codekansas/usa/releases/download/v1.0.0/language_navigation_ckpts.zip

# Downloads from the URL if the checkpoint directory is missing.
if [ ! -d $ckpts_paths ]; then
    echo "Downloading checkpoints from ${zip_url}"
    wget -O /tmp/language_navigation_ckpts.zip $zip_url
    unzip /tmp/language_navigation_ckpts.zip -d $EVAL_RUN_DIR
    rm /tmp/language_navigation_ckpts.zip
fi

export CKPTS_PATH_ROOT=$ckpts_paths

# Gets an empty evaluation directory under $EVAL_RUN_DIR.
eval_dir=${EVAL_RUN_DIR}/language-navigation/eval-$(date +%Y%m%d-%H%M%S)
mkdir -p $eval_dir
echo "Saving evaluation results to ${eval_dir}"

# Runs evaluations.
for eval_set in ${eval_sets[@]}; do
    echo "Running evaluation for ${eval_set}"

    eval_set_log_dir=$eval_dir/$eval_set

    if [ $use_slurm -eq 0 ]; then
        python -m ml.evaluation.evaluations.path_length ${eval_set} ${eval_set_log_dir}
        python -m ml.evaluation.evaluations.semantics ${eval_set} ${eval_set_log_dir}
    else
        slurm_log_dir=$eval_set_log_dir/logs
        mkdir -p $slurm_log_dir

        # Launches Python scripts on Slurm.
        sbatch --job-name=eval_path_length_${eval_set} \
            --output=$slurm_log_dir/path_length.%j.out \
            --error=$slurm_log_dir/path_length.%j.err \
            --time=1:00:00 \
            --mem=16G \
            --gres=gpu:1 \
            --cpus-per-task=4 \
            --partition=$slurm_partition \
            --wrap="python -m ml.evaluation.evaluations.path_length ${eval_set} ${eval_set_log_dir}"
        sbatch --job-name=eval_semantics_${eval_set} \
            --output $slurm_log_dir/semantics.%j.out \
            --error $slurm_log_dir/semantics.%j.err \
            --time=1:00:00 \
            --mem=16G \
            --gres=gpu:1 \
            --cpus-per-task=4 \
            --partition=$slurm_partition \
            --wrap="python -m ml.evaluation.evaluations.semantics ${eval_set} ${eval_set_log_dir}"

        # Sleep to avoid overburdening scheduler.
        sleep 1
    fi
done
