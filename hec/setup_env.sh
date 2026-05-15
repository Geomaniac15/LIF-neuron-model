#!/bin/bash
# One-time environment setup on the Lancaster HEC login node.
# Run from the login node: bash hec/setup_env.sh
#
# Creates a mamba env in your storage area so we don't blow the home-dir quota.
# Reference: https://lancaster-hec.readthedocs.io/en/latest/python.html

set -eo pipefail

# /etc/profile sets $global_storage and makes `module` available. `bash script.sh`
# starts a non-interactive shell that doesn't source it by default.
source /etc/profile

if [ -z "${global_storage:-}" ]; then
    echo "ERROR: \$global_storage is not set. Are you logged in to the HEC?" >&2
    exit 1
fi

# Redirect conda/mamba/pip caches off the small home partition (idempotent-ish:
# will fail loudly if the symlinks already exist, which is fine).
if [ ! -L "$HOME/.cache" ]; then
    mkdir -p "$global_storage/.cache" "$global_storage/.conda" "$global_storage/.mamba"
    ln -s "$global_storage/.cache" "$HOME/.cache"
    ln -s "$global_storage/.conda" "$HOME/.conda"
    ln -s "$global_storage/.mamba" "$HOME/.mamba"
fi

module add miniforge

ENV_PATH="$global_storage/lif_env"
if [ ! -d "$ENV_PATH" ]; then
    mamba create -y -p "$ENV_PATH" python=3.11 numpy matplotlib scikit-learn
else
    echo "Env already exists at $ENV_PATH — skipping create."
fi

source activate "$ENV_PATH"
python -c 'import numpy, matplotlib, sklearn; print("env ok:", numpy.__version__)'

echo
echo "Done. To use the env in a job script:"
echo "    module add miniforge"
echo "    source activate \$global_storage/lif_env"
