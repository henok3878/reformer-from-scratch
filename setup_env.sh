#!/bin/bash

# check environment if env name provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 [cpu|gpu]"
    echo "Example: $0 gpu"
    exit 1
fi

ENV_TYPE=$1

if [ "$ENV_TYPE" = "cpu" ]; then
    echo "Creating CPU environment..."
    conda env create -f reformer-cpu.yml
    conda activate reformer-cpu
    pip install -e .[dev]
    echo "Environment created! Activate with: conda activate reformer-cpu"
elif [ "$ENV_TYPE" = "gpu" ]; then
    echo "Creating GPU environment..."
    conda env create -f reformer-gpu.yml
    conda activate reformer-gpu
    echo "Environment created! Activate with: conda activate reformer-gpu"
else
    echo "Invalid option. Use 'cpu' or 'gpu'"
    exit 1
fi