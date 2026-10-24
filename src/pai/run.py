"""Run Physical ARC-AGI-3 (unofficial, inspired by ARC-AGI-3) in Genesis and record the video.

  python -m pai.run --video outputs/demo.mp4     # agent: Gemini if GEMINI_API_KEY is set, else offline heuristic
  python -m pai.run --offline --no-render        # fast logic check, no video

The agent decides; the robot only executes. After every action the cube poses are read from the
simulator, snapped to cells, and handed to the rules module (game.py), which returns the feedback
(tile colours, lit tiles, lamp) that is then displayed on the board.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import imageio
import numpy as np

from . import overlay
from .agents import AgentError, LOG, QuotaError, ReplayAgent, log, make_agent
from .game import Game, LEVELS
from .grid import CUBE, cell_center, stacks_from_poses
from .overlay import Hud
from .scene import Sim

FPS = 25
STEPS_PER_FRAME = 4              # 100 sim steps/s at 25 fps is 1x; --speed N renders one frame per 4N steps
OUT = Path(__file__).resolve().parents[2] / "outputs"


class Recorder:
    def __init__(self, path, hud):
        self.w = imageio.get_writer(path, fps=FPS, codec="libx264", quality=8, macro_block_size=1) if path else None
        self.hud, self.n = hud, 0

    def frame(self, rgb):
        if self.w:
            self.w.append_data(overlay.compose(rgb, self.hud))
            self.n += 1

    def raw(self, img, seconds):
        if self.w:
            for _ in range(int(seconds * FPS)):
                self.w.append_data(img)
                self.n += 1

    def t(self):
        """Seconds of video written so far."""
        return self.n / FPS

    def close(self):
        if self.w:
            self.w.close()


def snapshot(sim):
    """Observed layout from the physics: ({cell: [cubes bottom..top]}, held cube or None)."""
    poses, hand, held = sim.poses(), sim.hand_pos(), None
    for name, p in list(poses.items()):
        if p[2] > 0.06 and np.linalg.norm(p - hand) < 0.16:
            held = name
            del poses[name]
    return stacks_from_poses(poses), held


def execute(sim, game, action):
    """Carry out one action with the robot and return the observed (stacks, held)."""
    if action["name"] == "pick":
        sim.pick(action["object_id"])
    else:
        cell = tuple(action["cell"])
        stack = game.stacks.get(cell)
        if stack:                                    # stack on the real pose of the top cube
            p = sim.pos(stack[-1])
            xy, top = p[:2], p[2] + CUBE / 2
        else:
            xy, top = cell_center(cell), 0.0
        sim.place_at(game.held, xy, top)
    sim.hold(20)
    stacks, held = snapshot(sim)
    if action["name"] == "pick" and held != action["object_id"]:     # grasp failed: reset the arm
        sim.open_gripper()
        sim.go_home()
    return stacks, held


def sync_hud(hud, game, last=""):
    hud.actions, hud.lamp, hud.level = game.actions, game.solved, game.level.n
    hud.grid, hud.lit, hud.last = game.top_grid(), [list(c) for c in game.fb.lit], last


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="output mp4 (default: none)")
    ap.add_argument("--agent", default="llm-hypo", choices=["llm-plain", "llm-tricks", "llm-hypo"],
                    help="LLM condition when a key is set (default: llm-hypo)")
    ap.add_argument("--offline", action="store_true", help="use the offline heuristic even if a key is set")
    ap.add_argument("--replay", help="re-run the logged decisions of an earlier LLM run (agent_log.jsonl), no API calls")
    ap.add_argument("--no-render", action="store_true", help="skip rendering (fast logic test)")
    ap.add_argument("--speed", type=int, default=4, help="playback speed factor burned into the video (default 4)")
    ap.add_argument("--cap", type=int, default=40, help="max actions per level")
    a = ap.parse_args(argv)
    OUT.mkdir(exist_ok=True)
    if a.replay:
        agent = ReplayAgent(a.replay)
    else:
        LOG.unlink(missing_ok=True)
        agent = make_agent(a.agent, offline=a.offline)
    hud = Hud(agent=agent.label, speed=f"{a.speed}x speed")
    rec = Recorder(None if a.no_render else a.video, hud)
    sim = Sim(render_every=STEPS_PER_FRAME * a.speed, hook=None if a.no_render else rec.frame)
    game, results, tl = Game(1), [], {"title": 3.0, "level_complete": []}     # tl: video timestamps for screenshots
    rec.raw(overlay.title_card(agent.label), 6)
    for lv in LEVELS:
        tl[f"l{lv.n}_start"] = rec.t()
        game.start_level(lv.n)
        agent.reset()
        hud.hypothesis = "(ルールもゴールも知らされていない)"
        sync_hud(hud, game)
        sim.load_level(lv, game.fb)
        sim.hold(60)
        while not game.solved and game.actions < a.cap:
            try:
                d = agent.act(game.observation(), game.legal)
            except AgentError as e:
                print("agent failed:", e)
                break
            except QuotaError as e:
                print("STOPPED:", e)
                rec.close()
                return 3
            log({"agent": agent.label, "decision": d})
            hud.hypothesis, hud.note = d["hypothesis"], d.get("note", "")
            sim.hold(30)
            stacks, held = execute(sim, game, d["action"])
            effect = game.step(d["action"], observed=(stacks, held))
            sim.show(game.fb)                          # lamp / tiles come from the rules module
            sync_hud(hud, game, f"{game.history[-1].split(' -> ')[0]}  ->  {effect}"[:110])
            sim.hold(60 if effect != "no visible change" else 20)
            print(game.history[-1], "|", d["hypothesis"], "|", hud.note)
            m = re.search(r"仮説 (\d+)→(\d+)", hud.note)
            if lv.n == 2 and m and int(m[2]) < int(m[1]) and "l2_hypothesis_shrinks" not in tl:
                tl["l2_hypothesis_shrinks"] = rec.t() - 1.0     # during the next action, note still on screen
        ok = game.solved
        results.append({"level": lv.n, "solved": ok, "actions": game.actions})
        print(f"LEVEL {lv.n}: {'solved' if ok else 'NOT solved'} in {game.actions} actions")
        if ok:
            sim.hold(60)
            tl["level_complete"].append(rec.t() + 1.0)
            rec.raw(overlay.level_card(lv.n, len(LEVELS), game.actions, lv.optimal), 3)
        else:
            sim.hold(40)
    base = json.loads((OUT / "baseline.json").read_text()).get("random", {}) if (OUT / "baseline.json").exists() else {}
    rows = []
    for r, lv in zip(results, LEVELS):
        b = base.get("levels", {}).get(str(lv.n))
        rnd = f"{b['solve_rate'] * 100:.1f}% solved (cap {base['cap']})" if b else "n/a"
        rows.append((lv.n, r["actions"] if r["solved"] else None, rnd, lv.optimal))
    tl["end"] = rec.t() + 3.0
    tl["duration"] = rec.t() + 9.0
    rec.raw(overlay.table_card(agent.label, rows), 9)
    rec.close()
    (OUT / "demo_run.json").write_text(json.dumps({"agent": agent.label, "cap": a.cap, "speed": a.speed, "levels": results}, indent=1))
    (OUT / "demo_timeline.json").write_text(json.dumps(tl, indent=1))
    return 0 if all(r["solved"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
