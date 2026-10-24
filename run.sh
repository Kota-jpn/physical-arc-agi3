#!/usr/bin/env bash
# Usage: ./run.sh --video outputs/demo.mp4     (Gemini if GEMINI_API_KEY is set, else offline heuristic)
#        ./run.sh --offline --no-render        (fast logic check)
#        ./eval.sh --agent random|heuristic|llm  (symbolic evaluation, see README)
set -euo pipefail
cd "$(dirname "$0")"
[ -f .venv/bin/activate ] && source .venv/bin/activate
export PYTHONPATH=src
exec python -m pai.run "$@"
