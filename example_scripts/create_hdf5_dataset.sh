#!/bin/sh

script_path=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
root_path=$(dirname "$script_path")

mkdir -p "$root_path/datasets"


python $root_path/create_dataset.py \
--output-path="$root_path/datasets" \
--output-folder="electrostatic_poisson_1k" \
--min-seed=1 \
--max-seed=1000 \
--seed-step=100 \
--ntasks=2 \
--image-size=32 \
--conductive-cell-prob=0.5 \
--conductive-material-range=1,10 \
--max-iterations=2000 \
--save-states="first-10,interval-100"