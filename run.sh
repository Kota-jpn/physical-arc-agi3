#!/usr/bin/env bash
# Usage: ./run.sh --demo            -> outputs/demo.mp4
#        ./run.sh -i "<instruction>" [--disturb] [--video out.mp4]
set -euo pipefail
cd "$(dirname "$0")"
[ -f .venv/bin/activate ] && source .venv/bin/activate
export PYTHONPATH=src
exec python -m pai.run "$@"
