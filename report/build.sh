#!/usr/bin/env bash
# report.md -> report.html -> PDF (needs pandoc, ffmpeg, Google Chrome).
# The screenshots are cut from VIDEO at times taken from ../outputs/demo_timeline.json.
set -euo pipefail
cd "$(dirname "$0")"
VIDEO="${VIDEO:-../outputs/demo.mp4}"
mkdir -p img
read -r T_GUIDE T_L2 T_IMAG T_L4 < <(python3 -c "
import json; t = json.load(open('../outputs/demo_timeline.json'))
print(t['l1_start'] + 4, t['l2_hypothesis_shrinks'], t['imagination'], t['level_complete'][-1] + 3)")
take() { ffmpeg -loglevel error -y -ss "$2" -i "$VIDEO" -frames:v 1 -vf scale=1280:-1 "img/$1.png"; }
take f_guide "$T_GUIDE"; take f_l2 "$T_L2"; take f_imag "$T_IMAG"; take f_l4 "$T_L4"
pandoc report.md -s --shift-heading-level-by=-1 --toc --toc-depth=2 --number-sections --metadata toc-title="目次" --metadata pagetitle="説明資料_PAI最終課題_Kakinotane" --css style.css --embed-resources -o report.html
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
"$CHROME" --headless --disable-gpu --no-pdf-header-footer --print-to-pdf="説明資料_PAI最終課題_Kakinotane.pdf" "file://$PWD/report.html"
echo "built: $PWD/説明資料_PAI最終課題_Kakinotane.pdf"
