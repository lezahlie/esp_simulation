#!/bin/sh

script_path=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
root_path=$(dirname "$script_path")

python $root_path/process_dataset.py \
--dataset-path="$root_path/hdf5_dataset_example/electrostatic_poisson_32x32_1-1000.hdf5" \
--output-folder=simvp_dataset_example \
--normalize \
--simvp-format
