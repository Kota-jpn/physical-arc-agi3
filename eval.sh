#!/usr/bin/env bash
# Symbolic evaluation (no physics): ./eval.sh --agent random|heuristic|llm [--episodes N] [--cap N]
set -euo pipefail
cd "$(dirname "$0")"
[ -f .venv/bin/activate ] && source .venv/bin/activate
export PYTHONPATH=src
exec python -m pai.evaluate "$@"
