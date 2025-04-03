#!/bin/sh
div=$(printf '%*s' 100 '' | tr ' ' '-')

script_path=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
root_path=$(dirname "$script_path")

program_path="$root_path/create_dataset.py"

prog_args="
--output-folder=hdf5_dataset_example
--min-seed=1
--max-seed=1000
--seed-step=20
--ntasks=2
--image-size=32
--conductive-cell-prob=0.5
--conductive-material-range=1,10
--max-iterations=2000
--debug
"

echo "$div"

echo "$program_path"
for arg in $prog_args; do
    printf "%s\n" "$arg"
done

echo "$div"

python $program_path $prog_args