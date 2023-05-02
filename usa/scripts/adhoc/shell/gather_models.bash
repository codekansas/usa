#!/bin/bash

# Point this to whatever your model directory is
log_dir=$RUN_DIR
if [ ! -d $log_dir ]; then
    echo "Log directory $log_dir does not exist"
    exit 1
fi

# Specific hyperparameters.
datasets=(
    dataset_lab_stretch
    dataset_kitchen_stretch
    dataset_chess_stretch
    dataset_replica_apt_3_mnp
    dataset_studio_r3d
    dataset_lab_r3d
    dataset_chris_lab
)

eval_cache_file=/tmp/models_to_eval.txt
ls -1 $log_dir/*/run_*/checkpoints/ckpt.pt > ${eval_cache_file}

# Removes any cached stuff.
rm -rf $log_dir/*/run_*/eval_cache

# Directory to copy the checkpoints to.
out_dir=$EVAL_RUN_DIR/language-navigation-ckpts
rm -rf ${out_dir} && mkdir -p ${out_dir}

echo "Gathering models to ${out_dir}"

for dataset in ${datasets[@]}; do
    dataset_upper=$(echo $dataset | tr '[:lower:]' '[:upper:]')
    dataset_upper_trunc=$(echo $dataset_upper | cut -d '_' -f 1 --complement)
    dataset_lower_trunc=$(echo $dataset_upper_trunc | tr '[:upper:]' '[:lower:]')
    latest_model=$(cat ${eval_cache_file} | grep $dataset | sort -r | head -n 1)
    if [ -z "$latest_model" ]; then
        echo "Missing model for ${dataset}"
        continue
    fi

    latest_model_dir=$(dirname $(dirname $latest_model))
    out_dir_part=${out_dir}/${dataset_lower_trunc}
    mkdir -p ${out_dir_part}
    echo "Copying model and config file to ${dataset_lower_trunc}"
    cp $(realpath ${latest_model_dir}/checkpoints/ckpt.pt) ${out_dir_part}/ckpt.pt
    cp $(realpath ${latest_model_dir}/config.yaml) ${out_dir_part}/config.yaml
done

echo "Gathered models to ${out_dir}"

zip_path=${EVAL_RUN_DIR}/language_navigation_ckpts.zip
pushd ${EVAL_RUN_DIR}
zip -r ${zip_path} $(basename ${out_dir})
popd
echo "Zipped checkpoints to ${zip_path}"

echo "Done."
