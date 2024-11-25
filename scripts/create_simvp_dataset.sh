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
    "--seed-step=100"
    "--ntasks=2"
    "--image-size=32"
    "--max-iterations=2500"
    "--conductive-cell-prob=0.5"
    "--conductive-material-range=1,6"
    #"--conductive-cell-ratio=0.5"
    #"--conductive-material-count=6"
    #"--convergence-tolerance=1e-8"
    #"--enable-fixed-charges"
    #"--enable-absolute-permittivity"
    #"--disable-normalization"
    "--simvp-format"
    #"--plot-samples"
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