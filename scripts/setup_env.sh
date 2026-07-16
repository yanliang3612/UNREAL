#!/usr/bin/env bash

set -euo pipefail

ENV_NAME="${ENV_NAME:-unreal}"
PYTHON_VERSION="3.8.13"

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: Conda is not installed or is not available in PATH." >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "Conda environment '${ENV_NAME}' already exists; reusing it."
else
  conda create --yes --name "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

# Make `conda activate` available in this non-interactive shell.
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

conda install --yes \
  pytorch==1.12.1 \
  torchvision==0.13.1 \
  torchaudio==0.12.1 \
  cudatoolkit=11.3 \
  -c pytorch

python -m pip install torch_geometric torch-kmeans==0.2.0
python -m pip install \
  "https://data.pyg.org/whl/torch-1.12.0%2Bcu113/pyg_lib-0.3.1%2Bpt112cu113-cp38-cp38-linux_x86_64.whl" \
  "https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_cluster-1.6.0%2Bpt112cu113-cp38-cp38-linux_x86_64.whl" \
  "https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_scatter-2.1.0%2Bpt112cu113-cp38-cp38-linux_x86_64.whl" \
  "https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_sparse-0.6.16%2Bpt112cu113-cp38-cp38-linux_x86_64.whl" \
  "https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_spline_conv-1.2.1%2Bpt112cu113-cp38-cp38-linux_x86_64.whl"

echo
echo "Environment setup complete."
echo "Activate it with: conda activate ${ENV_NAME}"
