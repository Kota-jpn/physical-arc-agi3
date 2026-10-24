"""Symbolic evaluation (rules module only, no physics).

  python -m pai.evaluate --agent random|heuristic|llm-plain|llm-tricks|llm-hypo [--episodes 200] [--cap 200]

Results are merged into outputs/baseline.json under the agent's name.
"""
import argparse
import json
import statistics
from pathlib import Path

from .agents import HeuristicAgent, QuotaError, RandomAgent, make_agent, AgentError
from .game import Game, LEVELS

OUT = Path(__file__).resolve().parents[2] / "outputs" / "baseline.json"


def sequence(agent, cap, levels):  # raises QuotaError when the free-tier daily quota is used up
    """Play the levels in order with one agent (an LLM carries lessons from level to level).
    Returns {level: (actions used, solved within cap, LLM requests)}."""
    game, out = Game(levels[0]), {}
    for n in levels:
        game.start_level(n)
        agent.reset()
        req0 = getattr(agent, "requests", 0)
        while not game.solved and game.actions < cap:
            try:
                d = agent.act(game.observation(), game.legal)
            except AgentError:
                break
            game.step(d["action"])
        out[n] = (game.actions, game.solved, getattr(agent, "requests", 0) - req0)
        print(f"  level {n}: {'solved' if game.solved else 'NOT solved'} in {game.actions} actions", flush=True)
    return out


def evaluate(agent_factory, episodes, cap, levels):
    runs = []
    for i in range(episodes):
        print(f"episode {i + 1}/{episodes}", flush=True)
        try:
            runs.append(sequence(agent_factory(i), cap, levels))
        except QuotaError as e:
            print("STOPPED:", e, f"-- keeping the {len(runs)} finished episode(s)", flush=True)
            break
    if not runs:
        raise SystemExit(3)
    episodes = len(runs)
    res = {}
    for n in levels:
        r = [run[n] for run in runs]
        solved = [a for a, ok, _ in r if ok]
        res[str(n)] = {
            "optimal": LEVELS[n - 1].optimal,
            "solve_rate": len(solved) / episodes,
            "mean_actions_to_solve": round(statistics.mean(solved), 1) if solved else None,
            "mean_actions_unsolved_as_cap": round(statistics.mean(a if ok else cap for a, ok, _ in r), 1),
            "mean_llm_requests": round(statistics.mean(q for _, _, q in r), 1),
        }
    return res, episodes


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=["random", "heuristic", "llm-plain", "llm-tricks", "llm-hypo"], required=True)
    ap.add_argument("--episodes", type=int)
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3], help="subset for quick tests (not saved)")
    ap.add_argument("--cap", type=int, default=200)
    a = ap.parse_args(argv)
    if a.agent == "random":
        factory, label, episodes = (lambda i: RandomAgent(seed=i)), "random", a.episodes or 200
    elif a.agent == "heuristic":
        factory, label, episodes = (lambda i: HeuristicAgent()), "offline heuristic", a.episodes or 200
    else:
        if isinstance(make_agent(a.agent), HeuristicAgent):
            raise SystemExit("no GEMINI_API_KEY (or PAI_OFFLINE is set): the LLM agent is unavailable")
        factory, label, episodes = (lambda i: make_agent(a.agent)), make_agent(a.agent).label, a.episodes or 3
    lv_res, done = evaluate(factory, episodes, a.cap, a.levels)
    res = {"label": label, "episodes": done, "cap": a.cap, "levels": lv_res}
    if done < episodes:
        res["note"] = f"stopped by the free-tier daily quota after {done} of {episodes} episodes"
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if a.levels != [1, 2, 3]:
        return
    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    data.pop("llm", None)                       # old single-condition entry
    data[a.agent] = res
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    if done < episodes:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
