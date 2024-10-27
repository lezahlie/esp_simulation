#!/bin/bash
div=$(printf '%*s' 100 '' | tr ' ' '-')

script_path=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root_path=$(dirname "$script_path")

program_name=create_dataset

prog_args=(
    #"--output-path=path/to/dir"  
    "--output-folder=hdf5_dataset"   
    "--min-seed=1"                      
    "--max-seed=5000"
    "--seed-step=100"
    "--image-size=32"
    "--material-cell-ratio=0.75"
    "--conductive-material-ratio=0.25"
    "--max-iterations=2000"
    "--ntasks=2"
    "--plot-samples"
    #"--disable-normalization"
    #"--convergence-tolerance=1e-8"
    #"--enable-fixed-charges"
    #"--enable-absolute-permittivity"
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