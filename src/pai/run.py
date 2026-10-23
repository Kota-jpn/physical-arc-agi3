"""Closed-loop LLM-driven pick & place in Genesis.

  python -m pai.run --demo                       # build outputs/demo.mp4 (3 scenarios + cards)
  python -m pai.run -i "赤いキューブを青い皿の上に置いて" [--disturb] [--video out.mp4]
"""
import argparse
import sys
from pathlib import Path

import imageio
import numpy as np

from . import overlay
from .overlay import Hud
from .planner import Planner, PlanError, initial_on, pairs, validate, describe_state
from .scene import Sim, OBJECTS, PLATE_H, CUBE

SHORT = {"red_cube": "赤キューブ", "green_cube": "緑キューブ", "blue_cube": "青キューブ",
         "yellow_cube": "黄キューブ", "blue_plate": "青い皿"}
FPS = 25


def observe(sim):
    """Derive symbolic 'on' relations from simulator poses (ground truth of the closed loop)."""
    P = sim.poses()
    on, hand = {}, sim.hand_pos()
    for c, (kind, _, _) in OBJECTS.items():
        if kind != "cube":
            continue
        p, sup = P[c], "table"
        if p[2] > 0.06 and np.linalg.norm(p - hand) < 0.16:
            sup = "gripper"
        else:
            for o, q in P.items():
                if o == c:
                    continue
                r = 0.075 if OBJECTS[o][0] == "plate" else 0.03
                top = PLATE_H if OBJECTS[o][0] == "plate" else q[2] + CUBE / 2
                if np.linalg.norm(p[:2] - q[:2]) < r and abs(p[2] - CUBE / 2 - top) < 0.012:
                    sup = o
        on[c] = sup
    return on


class Recorder:
    def __init__(self, path, hud):
        self.w = imageio.get_writer(path, fps=FPS, codec="libx264", quality=8, macro_block_size=1) if path else None
        self.hud = hud

    def frame(self, rgb):
        if self.w:
            self.w.append_data(overlay.compose(rgb, self.hud))

    def raw(self, img, seconds):
        if self.w:
            for _ in range(int(seconds * FPS)):
                self.w.append_data(img)

    def close(self):
        if self.w:
            self.w.close()


def step_labels(plan):
    return [f"{SHORT[o]} → {SHORT[t]}" for o, t in pairs(plan)]


def show_plan(rec, planner, title, instruction, on, plan, failure=None):
    import json
    lines = [f"INSTRUCTION : {instruction}"]
    if failure:
        lines.append(f"FAILURE     : {failure}")
    lines += ["OBSERVED STATE (from simulator poses):"] + ["   " + l for l in describe_state(on).splitlines()]
    lines += ["", f"planner = {planner.backend_used}   ->   validated plan JSON:"]
    lines += ["   reason: " + plan.get("reason", "")]
    lines += ["   " + json.dumps(st, ensure_ascii=False) for st in plan["steps"]]
    rec.raw(overlay.text_card(title, lines, size=22), 6)


def run_task(sim, rec, hud, planner, instruction, disturb=False, max_replans=3):
    hud.instruction, hud.speed, hud.message_color = instruction, "", (255, 255, 255)
    hud.steps, hud.plan_tag = [], ""
    hud.message = "LLMにタスク分解を依頼中..."
    sim.hold(30)
    on = observe(sim)
    plan = planner.plan(instruction, on)
    hud.backend = planner.backend_used
    show_plan(rec, planner, "LLM task planning", instruction, on, plan)
    version, goals, done, replans, disturbed = 1, pairs(plan), [], 0, False
    log = [f"plan v1: {pairs(plan)}"]
    queue = list(pairs(plan))
    hud.plan_tag = "PLAN v1 (JSON → primitives)"
    hud.steps = [(l, "todo") for l in step_labels(plan)]
    hud.message = f"LLMの計画を検証OK ({len(queue)} 動作)"
    sim.hold(40)
    prior = 0   # number of hud rows belonging to earlier plans (kept for history)
    while queue:
        obj, tgt = queue.pop(0)
        idx = len(hud.steps) - len(queue) - 1
        # --- closed-loop check: have earlier achievements been disturbed? --------
        on = observe(sim)
        broken = [(o, t) for o, t in done if on.get(o) != t]
        failure = None
        if broken:
            failure = f"{broken[0][0]} is no longer on {broken[0][1]} (now on {on[broken[0][0]]})"
        else:
            try:
                validate({"steps": [{"action": "pick", "object": obj}, {"action": "place_on", "target": tgt}]}, on)
            except PlanError as e:
                failure = f"precondition violated: {e}"
        if failure is None:
            hud.steps[idx] = (hud.steps[idx][0], "doing")
            hud.message = f"実行: {SHORT[obj]} をつかむ"
            sim.pick(obj)
            on = observe(sim)
            if on[obj] != "gripper":
                failure = f"grasp of {obj} failed (it is on {on[obj]})"
            else:
                hud.message = f"実行: {SHORT[obj]} を {SHORT[tgt]} の上へ"
                top = PLATE_H if OBJECTS[tgt][0] == "plate" else sim.pos(tgt)[2] + CUBE / 2
                sim.place_at(obj, sim.pos(tgt)[:2], top)
                on = observe(sim)
                if on[obj] != tgt:
                    failure = f"{obj} is not on {tgt} after placing (it is on {on[obj]})"
        if failure is None:
            hud.steps[idx] = (hud.steps[idx][0], "done")
            hud.message = f"検証OK: {SHORT[obj]} は {SHORT[tgt]} の上にある"
            done.append((obj, tgt))
            sim.hold(25)
            if disturb and not disturbed and len(done) == 1 and queue:
                disturbed = True
                hud.message_color = (255, 170, 80)
                hud.message = "外乱を注入: 置いたキューブを手で払い落とす(意図的な失敗の再現)"
                p0, p1 = sim.pos(obj), np.array([OBJECTS[obj][2][0] + 0.1, 0.16, CUBE / 2])
                for k in range(1, 31):
                    sim.ent[obj].set_pos(p0 + (p1 - p0) * (k / 30) + [0, 0, 0.02 * np.sin(np.pi * k / 30)])
                    sim.step()
                sim.hold(40)
                hud.message_color = (255, 255, 255)
            continue
        # --- failure: close the loop through the LLM ----------------------------
        replans += 1
        if replans > max_replans:
            hud.message = "再計画の上限に到達"; sim.hold(30); return False, log
        if hud.steps and idx < len(hud.steps) and hud.steps[idx][1] != "done":
            hud.steps[idx] = (hud.steps[idx][0], "failed")
        hud.message_color = (255, 110, 110)
        hud.message = f"失敗検知: {failure}"
        log.append(f"failure: {failure}")
        sim.hold(50)
        hud.message_color = (255, 255, 255)
        hud.message = "観測した状態をLLMに渡して再計画..."
        if on.get(obj) == "gripper":
            sim.open_gripper(); sim.go_home()
        on = observe(sim)
        plan = planner.replan(instruction, {"steps": [{"action": "pick", "object": o} if k == 0 else {"action": "place_on", "target": t}
                                                      for o, t in goals for k in (0, 1)]},
                              on, None, failure)
        version += 1
        queue = list(pairs(plan))
        hud.plan_tag = f"RE-PLAN v{version} (観測状態から再計画)"
        hud.steps = [(l, "todo") for l in step_labels(plan)]
        done = [(o, t) for o, t in done if on.get(o) == t]
        log.append(f"replan v{version}: {pairs(plan)}")
        show_plan(rec, planner, f"LLM re-planning (v{version})", instruction, on, plan, failure)
        hud.message = f"再計画OK: {len(queue)} 動作"
        sim.hold(50)
    final = observe(sim)
    ok = all(final.get(o) == t for o, t in goals)
    hud.message_color = (120, 230, 140) if ok else (255, 110, 110)
    hud.message = "タスク完了: 目標状態を確認" if ok else "目標未達"
    sim.hold(75)
    return ok, log


DEMO = [
    dict(instruction="赤いキューブを青い皿の上に置いて、そのあと緑のキューブを赤の上に積んで", disturb=True,
         title="Scenario 1: 日本語指示 + 失敗からの再計画"),
    dict(instruction="Stack the blue cube on the green cube, then put the yellow cube on top of the blue one.",
         disturb=False, title="Scenario 2: English instruction (3-level tower)"),
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--instruction")
    ap.add_argument("--disturb", action="store_true")
    ap.add_argument("--video")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--no-render", action="store_true", help="skip rendering (fast logic test)")
    a = ap.parse_args(argv)
    out_dir = Path(__file__).resolve().parents[2] / "outputs"
    out_dir.mkdir(exist_ok=True)
    video = a.video or (str(out_dir / "demo.mp4") if a.demo else None)
    hud = Hud()
    rec = Recorder(None if a.no_render else video, hud)
    sim = Sim(hook=None if a.no_render else rec.frame)
    planner = Planner()
    jobs = DEMO if a.demo else [dict(instruction=a.instruction, disturb=a.disturb, title="Custom task")]
    if a.demo:
        rec.raw(overlay.card([("LLM × Genesis", 54, (255, 255, 255)),
                              ("自然言語で指示するロボットアーム:", 34, (150, 200, 255)),
                              ("検証付きLLMプランナーと、失敗からの閉ループ再計画", 34, (150, 200, 255)),
                              (f"Franka Panda / Genesis (CPU) / planner: {planner.backend}", 22, (180, 180, 180))]), 5)
    results = []
    for j in jobs:
        hud.title = j["title"]
        ok, log = run_task(sim, rec, hud, planner, j["instruction"], j["disturb"])
        results.append(ok)
        print(("OK  " if ok else "FAIL"), j["instruction"]); [print("   ", l) for l in log]
        if j is not jobs[-1]:
            sim.go_home(); sim.reset()
    if a.demo:
        rec.raw(overlay.card([("まとめ", 46, (255, 255, 255)),
                              ("LLM: 指示 → JSONプラン → シンボリック検証", 28, (150, 200, 255)),
                              ("Genesis: IK + 摩擦把持で実行 → 姿勢から成否を観測", 28, (150, 200, 255)),
                              ("失敗時は観測状態をLLMへ返して再計画", 28, (150, 200, 255))]), 4)
    rec.close()
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
