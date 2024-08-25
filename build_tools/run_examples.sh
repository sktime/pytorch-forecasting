#!/bin/bash

# Script to run all example notebooks.
# copy-paste from sktime's run_examples.sh
set -euxo pipefail

CMD="jupyter nbconvert --to notebook --inplace --execute --ExecutePreprocessor.timeout=600"

for notebook in docs/source/tutorials/*.ipynb; do
  echo "Running: $notebook"
  $CMD "$notebook"
done
