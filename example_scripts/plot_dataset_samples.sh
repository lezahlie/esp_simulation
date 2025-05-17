#!/bin/sh

script_path=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
root_path=$(dirname "$script_path")

plot_dataset()  {
    local num_samples=$1
    local dataset_path=$2

    python $root_path/process_dataset.py \
    --dataset-path="$dataset_path" \
    --sample-plots=$num_samples \
    --plot-states
}

plot_dataset 10 "$root_path/hdf5_dataset_example/electrostatic_poisson_32x32_1-1000.hdf5"
plot_dataset 10 "$root_path/hdf5_dataset_example/normalized_electrostatic_poisson_32x32_1-1000.hdf5"

