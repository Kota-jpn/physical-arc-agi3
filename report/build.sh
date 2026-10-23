#!/usr/bin/env bash
# Extract screenshots from outputs/demo.mp4 and render report.html -> PDF (needs ffmpeg + Google Chrome).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p img
for t in 7 22 24 50 76; do ffmpeg -loglevel error -y -ss $t -i ../outputs/demo.mp4 -frames:v 1 -vf scale=800:-1 img/shot_$t.png; done
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
"$CHROME" --headless --disable-gpu --no-pdf-header-footer --print-to-pdf="説明資料_PAI最終課題_Kakinotane.pdf" "file://$PWD/report.html"
