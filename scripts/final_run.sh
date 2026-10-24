#!/usr/bin/env bash
# Final Gemini run. Needs GEMINI_API_KEY in .env and unused free-tier quota (gemini-3.1-flash-lite: 500 requests/day).
# Rough cost: demo ~40 requests, llm-hypo eval ~150, llm-tricks eval ~150, llm-plain eval ~100 (< 500 in total).
# Run it in tmux:  tmux new-session -d -s final 'scripts/final_run.sh > outputs/final_run.log 2>&1'
# On a daily-quota stop (exit code 3) it keeps whatever finished and stops cleanly.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=src GEMINI_MODEL="${GEMINI_MODEL:-gemini-3.1-flash-lite}"
unset PAI_OFFLINE
mkdir -p outputs/old outputs/frames
stamp=$(date +%Y%m%d_%H%M%S)
save_log() { [ -f outputs/agent_log.jsonl ] && mv outputs/agent_log.jsonl "outputs/old/agent_log_$1_$stamp.jsonl"; return 0; }

# 0. keep the previous artefacts
save_log previous
[ -f outputs/demo.mp4 ] && cp outputs/demo.mp4 "outputs/old/demo_$stamp.mp4"
[ -f outputs/baseline.json ] && cp outputs/baseline.json "outputs/old/baseline_$stamp.json"
[ -f outputs/demo_run.json ] && cp outputs/demo_run.json "outputs/old/demo_run_$stamp.json"

# 1. live physics demo (llm-hypo) -> outputs/demo.mp4
echo "== 1. physics demo"
python -m pai.run --agent llm-hypo --cap 30 --speed 4 --video outputs/demo_new.mp4
rc=$?
if [ $rc -eq 3 ]; then echo "daily quota used up: stopping"; exit 3; fi      # 0 = all solved, 1 = some level not solved (kept honestly)
cp outputs/agent_log.jsonl outputs/agent_log_demo_hypo.jsonl
# fit the video into 60-180 s: re-render the same decisions at another speed (no API calls)
speed=$(python - <<'PY'
import json, math, subprocess
tl = json.load(open("outputs/demo_timeline.json"))
dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", "outputs/demo_new.mp4"]))
cards = 6 + 9 + 3 * len(tl["level_complete"])
if dur > 175:
    print(math.ceil(4 * (dur - cards) / (150 - cards)))
elif dur < 65:
    print(2)
else:
    print(0)
PY
)
if [ "$speed" != "0" ]; then
  echo "re-rendering at ${speed}x speed"
  python -m pai.run --replay outputs/agent_log_demo_hypo.jsonl --cap 30 --speed "$speed" --video outputs/demo_new.mp4
fi
mv outputs/demo_new.mp4 outputs/demo.mp4
ffprobe -v error -show_entries format=duration -of csv=p=0 outputs/demo.mp4

# 2. screenshots: title, L2 hypothesis-count moment, LEVEL COMPLETE card, end card
python - <<'PY'
import json, subprocess
tl = json.load(open("outputs/demo_timeline.json"))
l2 = tl.get("l2_hypothesis_shrinks", (tl["l2_start"] + tl.get("l3_start", tl["duration"])) / 2)
shots = {"title": tl["title"], "l2_hypothesis": l2, "end": tl["end"]}
if tl["level_complete"]:
    shots["level_complete"] = tl["level_complete"][0]
for name, t in shots.items():
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{t:.2f}", "-i", "outputs/demo.mp4", "-frames:v", "1", f"outputs/frames/{name}.png"], check=True)
    print(f"frame {name} at {t:.1f}s")
PY

# 3. symbolic evaluations (cap 40); each replaces its entry in outputs/baseline.json
for spec in "llm-hypo 3" "llm-tricks 3" "llm-plain 2"; do
  set -- $spec
  echo "== eval $1 ($2 episodes)"
  ./eval.sh --agent "$1" --episodes "$2" --cap 40 > "outputs/eval_$1.log" 2>&1
  rc=$?
  save_log "eval_$1"
  tail -n 3 "outputs/eval_$1.log"
  if [ $rc -eq 3 ]; then echo "daily quota used up: kept the finished episodes, stopping"; break; fi
done
echo "== done"
