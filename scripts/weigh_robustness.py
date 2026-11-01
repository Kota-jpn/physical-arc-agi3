"""How far off is the felt weight when the "real" arm differs from the imagined worlds?

  PYTHONPATH=src python scripts/weigh_robustness.py      # -> outputs/weigh_robustness.json

The imagined worlds always use the nominal model. The real arm gets one mismatch at a time: torque-sensor
noise, a weaker grip than modelled, or an unmodelled extra mass on the hand (e.g. a camera). Each cube of
level 4 is lifted from its start cell and weighed.
"""
import json
from pathlib import Path

import numpy as np

from pai.game import LEVELS, feedback, mass_of
from pai.imagine import Imagination
from pai.scene import Sim

OUT = Path(__file__).resolve().parents[1] / "outputs" / "weigh_robustness.json"
CASES = [("nominal", {}), ("torque noise 0.05 Nm", {"noise": 0.05}), ("torque noise 0.2 Nm", {"noise": 0.2}),
         ("grip 12 N (model 20 N)", {"grip": 12.0}), ("+0.1 kg on the hand", {"hand": 0.1})]


def main():
    sim = Sim(res=(320, 180))
    im = Imagination(sim, res=(320, 120))
    level, rng = LEVELS[3], np.random.default_rng(0)
    hand = sim.robot.get_link("hand")
    hand_mass = float(sim.robot.get_links_mass(links_idx_local=[hand.idx_local])[0])
    results = {}
    for name, case in CASES:
        errs = []
        for cube in level.start:
            sim.load_level(level, feedback(level, {c: [k] for k, c in level.start.items()}))
            if "hand" in case:
                sim.robot.set_links_mass(np.array([hand_mass + case["hand"]]), links_idx_local=[hand.idx_local])
            if "grip" in case:
                orig = sim.close_gripper
                sim.close_gripper = lambda steps=40, force=case["grip"]: orig(steps, force)
            sim.pick(cube)
            real = sim.felt + (rng.normal(0, case["noise"], 7) if "noise" in case else 0)
            felt, _, _ = im.weigh(*sim.before, real)
            errs.append(abs(felt - mass_of(level, cube)))
            print(f"{name:24s} {cube:7s} true {mass_of(level, cube):.2f} felt {felt:.3f}")
            sim.close_gripper = Sim.close_gripper.__get__(sim)
            if "hand" in case:
                sim.robot.set_links_mass(np.array([hand_mass]), links_idx_local=[hand.idx_local])
            sim.open_gripper()
            sim.go_home()
        results[name] = {"mean_abs_error_kg": round(float(np.mean(errs)), 3), "max_abs_error_kg": round(float(np.max(errs)), 3)}
    OUT.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print(json.dumps(results, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
