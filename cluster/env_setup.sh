#!/usr/bin/env bash
# Run once on the cluster login node to set up a reproducible environment.
#
# EDIT BEFORE RUNNING:
#   - the `module load` line: run `module avail cuda` on this cluster and
#     use whatever CUDA module actually exists.
#   - the torch index URL: run `nvidia-smi` and match the CUDA version it
#     reports (the cuXXX suffix in the index URL below).
set -euo pipefail

module load cuda/12.1 2>/dev/null || module load cuda 2>/dev/null || {
  echo "No cuda module loaded automatically -- run 'module avail cuda' and edit this script." >&2
}

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install triton

echo ""
echo "Environment ready. Activate it with:"
echo "  source $VENV_DIR/bin/activate"
