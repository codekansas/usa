#!/bin/bash

# ==========
# PARAMETERS
# ==========

# The eval set to run.
eval_set=lab_stretch
# eval_set=kitchen_stretch
# eval_set=chess_stretch
# eval_set=replica_apt_3_mnp
# eval_set=studio_r3d
# eval_set=lab_r3d
# eval_set=chris_lab

# The heights of the floor and ceiling.
floor_height=0.1
ceil_height=1.0

# The goal semantic locations.
semantics=(
    "The coffee maker"
    "The desks"
    "The person sitting at a computer"
)

# ==============
# END PARAMETERS
# ==============

# Kill script on first error.
set -e

ckpts_paths=$EVAL_RUN_DIR/language-navigation-ckpts
zip_url=https://github.com/codekansas/usa/releases/download/v1.0.0/language_navigation_ckpts.zip

# Downloads from the URL if the checkpoint directory is missing.
if [ ! -d $ckpts_paths ]; then
    echo "Downloading checkpoints from ${zip_url}"
    wget -O /tmp/language_navigation_ckpts.zip $zip_url
    unzip /tmp/language_navigation_ckpts.zip -d $EVAL_RUN_DIR
    rm /tmp/language_navigation_ckpts.zip
fi

# Gets an empty evaluation directory under $EVAL_RUN_DIR.
artifacts_dir=${EVAL_RUN_DIR}/language-navigation/eval-${eval_set}-$(date +%Y%m%d-%H%M%S)
mkdir -p $artifacts_dir
echo "Saving evaluation results to ${artifacts_dir}"

# Joins `goal_semantic_location` using a semicolon.
semantics_str=$(IFS=";"; echo "${semantics[*]}")

model_root=${ckpts_paths}/${eval_set}
ckpt_path=${model_root}/ckpt.pt
config_path=${model_root}/config.yaml

python -m ml.evaluation.evaluations.visualize_semantics \
    --floor-height ${floor_height} \
    --ceil-height ${ceil_height} \
    --goals "${semantics_str}" \
    --ckpt-path ${ckpt_path} \
    --config-path ${config_path} \
    --artifacts-dir ${artifacts_dir}
