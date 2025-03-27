#!/bin/bash
div=$(printf '%*s' 100 '' | tr ' ' '-')

script_path=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root_path=$(dirname "$script_path")

program_name=process_dataset

prog_args=(
    "--dataset-path=hdf5_dataset_example/electrostatic_poisson_32x32_1-100.hdf5"  
    "--output-folder=simvp_dataset_example"  
    "--simvp-format" 
    "--sample-plots=25"
    # "--plot-states"
    # "--output-path=path/to/dir"  
    # "--disable-normalization" 
    "--debug"
)


echo $div
echo "$program_name.py arguments:"
for arg in "${prog_args[@]}"; do
    echo "$arg"
done

echo $div
echo "Started creating shape maps"

python3 $root_path/$program_name.py ${prog_args[@]}

echo $div
echo "Finished creating shape maps"