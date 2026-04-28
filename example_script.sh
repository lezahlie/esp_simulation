#!/bin/sh

mkdir -p datasets

python create_dataset.py \
  --output-path="datasets" \
  --output-folder="electrostatic_poisson_1k" \
  --min-seed=1 \
  --max-seed=1000 \
  --seed-step=100 \
  --ntasks=2 \
  --image-size=32 \
  --max-iterations=2000 \
  --convergence-tolerance=1e-4 \
  --conductive-cell-prob=0.5 \
  --conductive-material-range=1,10 \
  --save-states="first-10,interval-100"

python process_dataset.py \
  --dataset-path="datasets/electrostatic_poisson_1k/electrostatic_poisson_32x32_1-1000.hdf5" \
  --normalize \
  --plot-states \
  --sample-plots=10

python process_dataset.py \
  --dataset-path="datasets/electrostatic_poisson_1k/normalized_electrostatic_poisson_32x32_1-1000.hdf5" \
  --plot-states \
  --sample-plots=10