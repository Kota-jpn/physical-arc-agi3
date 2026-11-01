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
from .grid import CUBE, cell_center, rgb, stacks_from_poses
from .imagine import LOADED, MASSES, Imagination
from .overlay import Hud
from .scene import CAM_FOV, CAM_LOOKAT, CAM_POS, Sim

FPS = 25
STEPS_PER_FRAME = 4              # 100 sim steps/s at 25 fps is 1x; --speed N renders one frame per 4N steps
OUT = Path(__file__).resolve().parents[2] / "outputs"


class Recorder:
    def __init__(self, path, hud):
        self.w = imageio.get_writer(path, fps=FPS, codec="libx264", quality=None, pixelformat="yuv420p",
                                    macro_block_size=8, ffmpeg_params=["-crf", "16", "-preset", "slow"]) if path else None
        self.hud, self.n, self.last = hud, 0, None
        self.marks = []                            # [name, start s, length s] of each card (for the narration)
        self.seconds = {}                          # card name -> length override (--cards)

    def frame(self, rgb):
        if self.w:
            self.last = overlay.compose(rgb, self.hud)
            self.w.append_data(self.last)
            self.n += 1

    def raw(self, img, seconds, name=None):
        seconds = max(seconds, self.seconds.get(name, 0))      # never shorter than the reading time
        if name:
            self.marks.append([name, round(self.t(), 2), seconds])
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
        sim.hold(20)
        stacks, held = snapshot(sim)
        if held != action["object_id"]:                  # grasp failed: reset the arm
            sim.open_gripper()
            sim.go_home()
        return stacks, held
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
    return snapshot(sim)


def show_imagination(rec, im, cube, mass, best, taus, real):
    """Play the imagined worlds in the camera area, then hold the comparison with the real torques."""
    if not rec.w or not im.frames:
        return
    res = np.asarray(im.frames[0]).shape[1], np.asarray(im.frames[0]).shape[0]
    pos, lookat, fov = im.cam_pose
    xs = [overlay.project((o[0] + 0.5, o[1], 0.25), pos, lookat, fov, res=res)[0] for o in im.offsets]
    for k, view in enumerate(im.frames):
        rec.raw(overlay.imagine_frame(rec.last, view, MASSES, xs, cube=cube), 1 / FPS, None if k else "imagination")
    rec.raw(overlay.imagine_frame(rec.last, im.frames[-1], MASSES, xs, taus[:, LOADED], real[LOADED], best, mass, cube), 3)


COLOUR_JA_ARC = {1: "青", 2: "赤", 3: "緑", 4: "黄", 5: "灰", 6: "マゼンタ", 7: "オレンジ", 8: "水色", 9: "栗色", 0: "黒", None: "なし"}
RULES = {1: "キューブを同じ色のタイルに置く (黄色のキューブはおとり)",
         2: "オレンジのスイッチに何か載っている間だけ灰色のタイルの本当の色が見える。その色に合わせて置く",
         3: "マゼンタの印の数 (3 個) と同じ高さの塔を、水色のタイルの上に作る",
         4: "天秤。2 つの栗色の皿が釣り合うと点く。青 0.3 kg = 緑 0.1 + 黄 0.2 (赤は 0.04)"}


def sync_hud(hud, game):
    hud.actions, hud.lamp, hud.level, hud.optimal = game.actions, game.solved, game.level.n, game.level.optimal
    hud.grid, hud.lit, hud.felt = game.top_grid(), [list(c) for c in game.fb.lit], dict(game.felt)


def target_cell(game, action):
    """Board cell the action touches (the cube's cell for pick), for the HUD highlight."""
    if action["name"] == "place":
        return list(action["cell"])
    return next((list(c) for c, st in game.stacks.items() if action["object_id"] in st), None)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="output mp4 (default: none)")
    ap.add_argument("--agent", default="llm-hypo", choices=["llm-plain", "llm-tricks", "llm-hypo"],
                    help="LLM condition when a key is set (default: llm-hypo)")
    ap.add_argument("--offline", action="store_true", help="use the offline heuristic even if a key is set")
    ap.add_argument("--replay", help="re-run the logged decisions of an earlier LLM run (agent_log.jsonl), no API calls")
    ap.add_argument("--no-render", action="store_true", help="skip rendering (fast logic test)")
    ap.add_argument("--speed", default="4", help="playback speed burned into the video: N, or one per level as N,N,N (default 4)")
    ap.add_argument("--cap", type=int, default=40, help="max actions per level")
    ap.add_argument("--cards", help="JSON {card name: seconds} overriding card lengths (to fit a narration)")
    ap.add_argument("--levels", type=int, nargs="+", default=[lv.n for lv in LEVELS], help="levels to play (quick checks)")
    ap.add_argument("--remark", action="append", default=[], metavar="N=TEXT",
                    help="extra line on the LEVEL N clear card, e.g. to say how a replayed run was solved")
    a = ap.parse_args(argv)
    OUT.mkdir(exist_ok=True)
    if a.replay:
        agent = ReplayAgent(a.replay)
    else:
        LOG.unlink(missing_ok=True)
        agent = make_agent(a.agent, offline=a.offline)
    speeds = [int(v) for v in a.speed.split(",")]
    speeds += speeds[-1:] * (len(LEVELS) - len(speeds))
    hud = Hud(agent=agent.label, n_levels=len(LEVELS))
    rec = Recorder(None if a.no_render else a.video, hud)
    if a.cards:
        rec.seconds = json.loads(Path(a.cards).read_text())
    sim = Sim(res=overlay.CAM_RES, render_every=STEPS_PER_FRAME * speeds[0], hook=None if a.no_render else rec.frame)
    im = Imagination(sim, res=(overlay.CAM_RES[0], 540))
    shown = 0                                       # imagination clips played in full (weight level)
    game, results, tl = Game(a.levels[0]), [], {"title": 3.0, "level_complete": []}     # tl: video timestamps for screenshots
    backdrop = None
    if rec.w:                                       # a still of the first board behind the title
        sim.hook, first = None, Game(a.levels[0])
        sim.load_level(first.level, first.fb)
        backdrop, sim.hook = sim.cam.render()[0], rec.frame
    for img, sec, name in [(overlay.title_card(backdrop), 5, "title"), (overlay.arc_card(), 8, "arc"),
                           (overlay.rules_card(), 7, "rules"), (overlay.method_card(), 9, "method")]:
        rec.raw(img, sec, name)
    for lv, speed in zip(LEVELS, speeds):
        if lv.n not in a.levels:
            continue
        if lv.kind == "balance":
            rec.raw(overlay.physics_card(), 8, "physics")
        rec.raw(overlay.level_intro(lv.n, len(LEVELS)), 3, f"l{lv.n}_intro")
        tl[f"l{lv.n}_start"] = rec.t()
        sim.render_every, hud.speed = STEPS_PER_FRAME * speed, f"{speed}倍速" if speed > 1 else ""
        game.start_level(lv.n)
        agent.reset()
        hud.hypothesis = "(まだ何も知らない。ルールもゴールも教えられていない)"
        hud.action = hud.effect = hud.note = ""
        hud.target, hud.callouts = None, []
        sync_hud(hud, game)
        sim.load_level(lv, game.fb)
        sim.hold(60)
        if lv.n == 1 and rec.last is not None:
            rec.raw(overlay.guide_card(rec.last), 8, "guide")
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
            hud.action, hud.effect, hud.target = overlay.ja_action(d["action"]), "", target_cell(game, d["action"])
            hud.callouts, before = [], dict(game.fb.floor)
            sim.hold(30)
            stacks, held = execute(sim, game, d["action"])
            felt = None
            if d["action"]["name"] == "pick" and held == d["action"]["object_id"]:     # weigh by imagination
                full = lv.kind == "balance" and shown < 2
                felt, best, taus = im.weigh(*sim.before, sim.felt, record=full and rec.w is not None)
                if lv.kind == "balance" and rec.w:
                    show_imagination(rec, im, held, felt, best, taus, sim.felt)
                    tl.setdefault("imagination", rec.t() - 1.5)       # the comparison is on screen
                    shown += full
            effect = game.step(d["action"], observed=(stacks, held), felt=felt)
            sim.show(game.fb)                          # lamp / tiles come from the rules module
            sync_hud(hud, game)
            hud.effect = overlay.ja_effect(effect)
            if felt is not None:
                hud.effect = f"重さ {game.felt[held]:.2f} kg" + ("" if hud.effect == "変化なし" else "、" + hud.effect)
            for cell, colour in game.fb.floor.items():     # a tile that changed colour may sit under a cube
                if before.get(cell) != colour:
                    x, y = overlay.project((*cell_center(cell), 0.0), CAM_POS, CAM_LOOKAT, CAM_FOV)
                    hud.callouts.append((x, y, f"{COLOUR_JA_ARC[before.get(cell)]} → {COLOUR_JA_ARC[colour]}", rgb(colour)))
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
            remark = next((r.split("=", 1)[1] for r in a.remark if r.split("=", 1)[0] == str(lv.n)), "")
            rec.raw(overlay.level_card(lv.n, game.actions, lv.optimal, RULES[lv.n], hud.hypothesis, remark), 6 + 2 * bool(remark), f"l{lv.n}_clear")
        else:
            sim.hold(40)
            hud.failed.add(lv.n)
            rec.raw(overlay.level_card(lv.n, game.actions, lv.optimal, RULES[lv.n], hud.hypothesis, solved=False), 5, f"l{lv.n}_fail")
    data = json.loads((OUT / "baseline.json").read_text()) if (OUT / "baseline.json").exists() else {}
    base, rows = data.get("random", {}), []
    for r in results:
        lv = LEVELS[r["level"] - 1]
        b = base.get("levels", {}).get(str(lv.n))
        rnd = f"{base['cap']} 手以内に解けたのは {b['solve_rate'] * 100:.1f}%" if b else "-"
        rows.append((lv.n, r["actions"] if r["solved"] else None, rnd, lv.optimal))
    tl["end"] = rec.t() + 3.0
    tl["duration"] = rec.t() + 7.0
    k = sum(r["solved"] for r in results)
    with_w, without_w = (data.get(key, {}).get("levels", {}).get("4") for key in ("llm-hypo", "llm-hypo-noweight"))
    ablation = (f"レベル 4 を物理なしで繰り返すと、重さを感じられるとき {with_w['solve_rate'] * with_w['episodes']:.0f}/{with_w['episodes']} 回、"
                f"感じられないとき {without_w['solve_rate'] * without_w['episodes']:.0f}/{without_w['episodes']} 回クリアした。"
                if with_w and without_w else "これは 1 回分の実行例。でたらめな動きは 200 回試した結果。")
    rec.raw(overlay.table_card(rows, [f"ルールを知らされないまま、試行と観察だけで {k} / {len(LEVELS)} レベルをクリアした。", ablation,
                                      "見た目で分からない性質を触って確かめてから決める。荷物の仕分けや積み付けにつながる。"]), 7, "end")
    rec.close()
    (OUT / "demo_run.json").write_text(json.dumps({"agent": agent.label, "cap": a.cap, "speed": speeds, "levels": results}, indent=1))
    tl["cards"] = rec.marks
    (OUT / "demo_timeline.json").write_text(json.dumps(tl, indent=1))
    return 0 if all(r["solved"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
