#!/bin/sh
div=$(printf '%*s' 100 '' | tr ' ' '-')

script_path=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
root_path=$(dirname "$script_path")

program_path="$root_path/process_dataset.py"

prog_args="
--dataset-path=hdf5_dataset_example/electrostatic_poisson_32x32_1-1000.hdf5
--sample-plots=25
--debug
"

echo "$div"

echo "$program_path"
for arg in $prog_args; do
    printf "%s\n" "$arg"
done

echo "$div"

python $program_path $prog_args