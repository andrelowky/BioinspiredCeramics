#!/usr/bin/env bash
# Set up a laptop (Linux or WSL) to run the agents against models on Azure, with every tool on the CPU.
# Run it from anywhere; it works from the repository root.
#
# Builds two environments outside the repository, since imports from a Windows folder (/mnt/c) are slow:
#   $BIOCERAMICS_VENV     (default ~/.venvs/bioceramics)     the pipeline, with JAX for the CPU
#   $BIOCERAMICS_MD_VENV  (default ~/.venvs/bioceramics-md)  PyTorch for the CPU and MACE, for md_bulk of silicon
#                                                            carbide, as ~/simtools/env does on the Spark
# PYTHON names the interpreter, Python 3.12 or newer with venv and ensurepip (on WSL, ~/miniconda3/bin/python
# when the system Python lacks ensurepip). With certbundle.crt in the repository root, pip trusts it.
set -euo pipefail
cd "$(dirname "$0")/../.."
VENV=${BIOCERAMICS_VENV:-$HOME/.venvs/bioceramics}
MD_VENV=${BIOCERAMICS_MD_VENV:-$HOME/.venvs/bioceramics-md}
PYTHON=${PYTHON:-python3}
if [ -f certbundle.crt ]; then
  export PIP_CERT=${PIP_CERT:-$PWD/certbundle.crt} SSL_CERT_FILE=${SSL_CERT_FILE:-$PWD/certbundle.crt}
fi
"$PYTHON" -c 'import sys; assert sys.version_info >= (3, 12), f"Python 3.12 or newer is needed, not {sys.version}"'

# requirements.txt, with JAX for the CPU on Linux as well. The FEM tool runs on the CPU anyway.
"$PYTHON" -m venv "$VENV"
grep -v 'jax\[cuda' requirements.txt | sed 's/; sys_platform != "linux"//' > "$VENV/requirements-cpu.txt"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q -r "$VENV/requirements-cpu.txt"

# MACE for md_bulk of silicon carbide. Without it a silicon carbide design stops the run at the judge.
# python -m tools.md.bulk imports the whole tools package, JAX included, so this environment has the pipeline's
# requirements too.
"$PYTHON" -m venv "$MD_VENV"
"$MD_VENV/bin/pip" install -q --upgrade pip
"$MD_VENV/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cpu
"$MD_VENV/bin/pip" install -q -r "$VENV/requirements-cpu.txt" mace-torch

"$VENV/bin/python" -m pipeline.benchmark.make_charts
echo "Environments ready: $VENV and $MD_VENV. Run commands through pipeline/scripts/laptop.sh."
