#!/bin/bash
# One-time environment setup on the Lancaster Hex cluster (UCREL).
# Run from the login node: bash hex/setup_env.sh
#
# Hex doesn't have the HEC's module system or $global_storage. We just use
# a venv in the project directory.

set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PATH="$PROJECT_DIR/.venv"

cd "$PROJECT_DIR"

if [ ! -d "$VENV_PATH" ]; then
    echo "Creating venv at $VENV_PATH"
    python3 -m venv "$VENV_PATH"
else
    echo "Venv already exists at $VENV_PATH - skipping create."
fi

source "$VENV_PATH/bin/activate"

pip install --upgrade pip
pip install numpy matplotlib scikit-learn

python -c 'import numpy, matplotlib, sklearn; print("env ok:", numpy.__version__)'

echo
echo "Done. To use the env in a job script:"
echo "    source $VENV_PATH/bin/activate"
