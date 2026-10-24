#!/usr/bin/env bash
# Prepare the screenshots and render report.html -> PDF (needs ffmpeg + Google Chrome).
#
# Screenshots: if ../outputs/frames/{title,l2_hypothesis,level_complete,end}.png exist (written by
# scripts/final_run.sh) they are used. Otherwise each one is cut from VIDEO at the times below.
#
# --- edit these to re-take the screenshots from a video ------------------------------------------
VIDEO="${VIDEO:-../outputs/demo.mp4}"      # fallback: ../outputs/demo_offline.mp4 if demo.mp4 is missing
T_TITLE=2                 # seconds: title card
T_L2=24                   # seconds: L2 in progress (switch revealed the targets)
T_COMPLETE=16             # seconds: LEVEL COMPLETE card
T_END_FROM_END=4          # result card = this many seconds before the end of the video
# -----------------------------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"
FR=../outputs/frames
[ -f "$VIDEO" ] || VIDEO=../outputs/demo_offline.mp4
mkdir -p img
DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$VIDEO" 2>/dev/null || echo 0)
T_END=$(awk -v d="$DUR" -v e="$T_END_FROM_END" 'BEGIN{printf "%.2f", d-e}')
# usage: take <out name> <frames/ name> <timestamp>
take() {
  if [ -f "$FR/$2.png" ]; then
    ffmpeg -loglevel error -y -i "$FR/$2.png" -vf scale=800:-1 "img/$1.png"
  else
    echo "no $FR/$2.png: cutting from $VIDEO at $3 s" >&2
    ffmpeg -loglevel error -y -ss "$3" -i "$VIDEO" -frames:v 1 -vf scale=800:-1 "img/$1.png"
  fi
}
take f_title title "$T_TITLE"; take f_l2 l2_hypothesis "$T_L2"
take f_complete level_complete "$T_COMPLETE"; take f_end end "$T_END"
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
"$CHROME" --headless --disable-gpu --no-pdf-header-footer --print-to-pdf="説明資料_PAI最終課題_Kakinotane.pdf" "file://$PWD/report.html"
echo "built: $PWD/説明資料_PAI最終課題_Kakinotane.pdf"
