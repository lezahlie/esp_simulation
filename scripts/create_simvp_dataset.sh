#!/bin/bash
div=$(printf '%*s' 100 '' | tr ' ' '-')

script_path=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root_path=$(dirname "$script_path")

program_name=create_dataset

prog_args=(
    #"--output-path=path/to/dir"  
    "--output-folder=simvp_dataset"   
    "--min-seed=1"                      
    "--max-seed=1000"
    "--seed-step=50"
    "--image-size=32"
    "--material-cell-ratio=1.0"
    "--conductive-material-ratio=0.5"
    "--max-iterations=2000"
    "--ntasks=2"
    "--simvp-format"
    #"--convergence-tolerance=1e-7"
    #"--plot-samples"
    #"--disable-normalization"
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