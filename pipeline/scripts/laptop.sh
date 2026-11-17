#!/usr/bin/env bash
# Run one command in the laptop environments that setup_laptop.sh built, from the repository root:
#   bash pipeline/scripts/laptop.sh python -m pipeline.run "question" data/corpus
# Trusts certbundle.crt in the repository root, when it is there, for a network that re-signs TLS. The model
# endpoints and keys come from .env (see .env.example).
set -euo pipefail
cd "$(dirname "$0")/../.."
VENV=${BIOCERAMICS_VENV:-$HOME/.venvs/bioceramics}
if [ -f certbundle.crt ]; then
  export SSL_CERT_FILE=${SSL_CERT_FILE:-$PWD/certbundle.crt} REQUESTS_CA_BUNDLE=${REQUESTS_CA_BUNDLE:-$PWD/certbundle.crt}
fi
# md_bulk of silicon carbide runs MACE in the second environment.
export SIMTOOLS_PYTHON=${SIMTOOLS_PYTHON:-${BIOCERAMICS_MD_VENV:-$HOME/.venvs/bioceramics-md}/bin/python}
export PATH="$VENV/bin:$PATH"
# A Claude Code session exports its own ANTHROPIC_BASE_URL, which must not reach the run.
exec env -u ANTHROPIC_BASE_URL "$@"
