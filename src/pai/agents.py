"""Agents: random, offline heuristic explorer, and the Gemini LLM agent.

An agent has `label` and `act(obs, legal) -> {"hypothesis": str, "action": {...}}`, where `obs` is
`Game.observation()` and `legal(action)` returns None or the reason an action is illegal.
"""
import json
import os
import random
import time
from pathlib import Path

from .game import CUBES, check
from .grid import ARC, N

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "outputs" / "agent_log.jsonl"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
JA = {0: "黒", 1: "青", 2: "赤", 3: "緑", 4: "黄", 5: "灰色", 6: "マゼンタ", 7: "オレンジ", 8: "水色", 9: "栗色"}


def log(record):
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def pick(cube):
    return {"name": "pick", "object_id": cube}


def place(cell):
    return {"name": "place", "cell": list(cell)}


# ------------------------------------------------------------------------------ random
class RandomAgent:
    label = "random"

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def reset(self):
        pass

    def act(self, obs, legal):
        while True:
            if obs["held"]:
                a = place((self.rng.randrange(N), self.rng.randrange(N)))
            else:
                a = pick(self.rng.choice(list(CUBES)))
            if legal(a) is None:
                return {"hypothesis": "ランダムに動かしてみる。", "action": a}


# ------------------------------------------------------------------------------ offline heuristic
class HeuristicAgent:
    """Systematic explorer: try cubes on the marked tiles, note what changed, then exploit.

    It encodes three generic hypotheses (same colour matches; something reveals hidden tiles;
    stacking count relates to the markers) and is NOT an LLM. Probing order prefers one-of-a-kind
    tiles over repeated ones (a repeated row of squares looks like a counter, not a button)."""
    label = "offline heuristic"

    def __init__(self):
        self.level = None

    def reset(self):
        self.known = {}            # cell -> colour of every marked tile seen so far
        self.tried = set()         # (cube, cell) probes already made
        self.switch = self.counter = None
        self.prev = self.last_place = None
        self.pending = None        # cell for the place that follows the current pick

    def act(self, obs, legal):
        if obs["level"] != self.level:
            self.level = obs["level"]
            self.reset()
        self._learn(obs)
        action, why = self._decide(obs)
        if legal(action) is not None:                 # never send an impossible action
            action, why = self._fallback(obs, legal), "計画が実行できないので、別の動きを試す。"
        self.prev = obs
        if action["name"] == "place":
            self.last_place = tuple(action["cell"])
        else:
            self.last_place = None
        return {"hypothesis": why, "action": action}

    def _learn(self, obs):
        for r in range(N):
            for c in range(N):
                col = obs["floor"][r][c]
                if col and (col != 5 or (r, c) not in self.known):
                    self.known[(r, c)] = col
        if self.last_place is None or self.prev is None:
            return
        cell = self.last_place
        floor_changed = any(obs["floor"][r][c] != self.prev["floor"][r][c]
                            for r in range(N) for c in range(N) if (r, c) != cell)
        gained = [tuple(x) for x in obs["lit"] if x not in self.prev["lit"]]
        if floor_changed and self.switch is None:
            self.switch = cell
        elif gained and all(g != cell for g in gained) and self.counter is None:
            self.counter = cell

    def _decide(self, obs):
        if obs["held"] and self.pending:
            cell, self.pending = self.pending[0], None
            return place(cell), self.pending_why
        cubes = {c["id"]: c for c in obs["cubes"]}
        free = [c for c in cubes.values() if c["cell"] is not None]
        height = lambda cell: sum(1 for c in free if c["cell"] == list(cell))

        def go(cube, cell, why):
            if obs["held"]:
                return place(cell), why
            self.pending, self.pending_why = (cell,), why
            return pick(cube), why

        if obs["held"]:                                # holding without a plan (after a fallback)
            return place(next(iter(self.known), (0, 0))), "手に持ったキューブをとりあえず置く。"
        # 1. stacking changes the markers: build the tower on that tile
        if self.counter:
            cube = next(c["id"] for c in free if c["cell"] != list(self.counter))
            return go(cube, self.counter, "積んだ数とマーカーの点灯数が連動している。マーカーの数だけ積めば点灯するはず。")
        # 2. same-colour pairs (tiles seen so far, including ones revealed earlier)
        colour_cube = {c["colour"]: c for c in free}
        for cell, col in sorted(self.known.items()):
            c = colour_cube.get(col)
            if cell != self.switch and c and not (c["cell"] == list(cell) and c["height"] == 0):
                why = f"同じ色のタイルに置くと光りそうだ。{JA[col]}キューブを{JA[col]}のタイルへ。"
                if self.switch:
                    why = f"スイッチで見えた{JA[col]}のタイルに{JA[col]}キューブを合わせる。"
                return go(c["id"], cell, why)
        # 3. a switch is known but nothing rests on it: press it with a cube that no tile needs
        if self.switch and height(self.switch) == 0:
            needed = set(self.known.values())
            spare = next((c["id"] for c in free if c["colour"] not in needed), free[0]["id"])
            return go(spare, self.switch, "スイッチにキューブを置くと灰色のタイルが色を現す。余ったキューブで押す。")
        # 4. probe: a not-yet-tried cube on the rarest marked tile
        counts = {}
        for col in self.known.values():
            counts[col] = counts.get(col, 0) + 1
        for cell in sorted(self.known, key=lambda x: (counts[self.known[x]], x)):
            for c in free:
                if (c["id"], cell) not in self.tried:
                    self.tried.add((c["id"], cell))
                    return go(c["id"], cell, f"{JA[c['colour']]}キューブを{JA[self.known[cell]]}のタイルに置いたら何か変わるか試す。")
        return self._fallback(obs, None), "手がかりがないのでランダムに試す。"

    def _fallback(self, obs, legal):
        return RandomAgent().act(obs, legal or (lambda a: None))["action"]


# ------------------------------------------------------------------------------ Gemini (OpenAI-compatible endpoint)
# Three prompt-only conditions (inference only, no training):
#   plain  : the LLM sees the board + experiment memory and answers one action per request
#   tricks : plain + frame diff, dead-action memory, repeat/revisit guards, 1-4 action batches, notes memory
#   hypo   : tricks + hypotheses written as code, falsified locally, and discriminating experiments (hypo.py)
BASE = """You are the brain of a robot arm that is dropped into an unknown turn-based puzzle on a {n}x{n} grid board seen from above.
You are NOT told the rules or the goal. You must explore, infer the rules, and then act efficiently.
Facts you are given:
- Colours are ARC integers: {palette}.
- Cells are [row, col], 0-indexed. Row 0 is nearest the robot. Movable objects are solid-colour cubes; a cube's id is its colour name.
- Actions: {{"name":"pick","object_id":"<cube id>"}} lifts a top cube (gripper must be empty);
  {{"name":"place","cell":[row,col]}} puts the held cube on a cell; placing on an occupied cell stacks (max height {h}).
- Some unknown condition completes the level; then the lamp turns on. Later levels may use different rules.
- Every executed action is counted. Solve each level in as few actions as possible.
How to work:
- Each turn you get the board, the experiment memory (every action in this level and what visibly changed afterwards) and lessons from earlier levels, which may or may not carry over.
- Explore broadly first: interact at least once with each visibly distinct kind of marked cell before settling on a theory.
- Keep 2-3 competing hypotheses about what makes the lamp turn on. Prefer actions whose outcome would tell the hypotheses apart; once a hypothesis is confirmed, act on it directly without further tests.
- Look at everything that changes after an action (any cell, any colour, any border, the lamp), not only the cell you touched, and compare it with what you expected.
- If two actions under the same hypothesis produced no new visible change, drop that hypothesis and switch to a different one. Never repeat an action that already had no effect.
"""
PLAIN_REPLY = """Reply with JSON only:
{{"observations": "<what changed and what it suggests>", "hypotheses": ["<candidate 1>", "<candidate 2>"], "test": "<which hypothesis the action tests>",
 "hypothesis": "<one short Japanese sentence: your current best guess about the rule>", "lesson": "<one short English sentence: what you know so far about how this world works>",
 "action": {{...}}}}"""
TRICKS_RULES = """- You may return a batch of 1-4 actions to save requests (they are executed in order). Mark each action with "expect_change": true if you expect a visible change
  (a tile colour, a border, the lamp) after it, else false; the batch is stopped early if reality differs. Use a batch only when the steps are clear (e.g. pick then place).
- You also get a diff of what changed after your last action, a list of placements that were already tried without any visible effect, and your own notes.
  Notes: rewrite your short notes (max 60 words) whenever you learn something; they are carried to the next level.
"""
TRICKS_REPLY = """Reply with JSON only:
{{"observations": "<what changed and what it suggests>", "hypotheses": ["<candidate 1>", "<candidate 2>"],
 "hypothesis": "<one short Japanese sentence: your current best guess about the rule>", "notes": "<your notes>",
 "actions": [{{"name": "pick", "object_id": "...", "expect_change": false}}, {{"name": "place", "cell": [r, c], "expect_change": true}}]}}"""
HYPO_RULES = """- Besides actions, give 2-4 competing hypotheses as short Python functions `def lit(state): ...` that return True exactly when the lamp is ON in a board state.
  Code checks them against every state observed so far and discards the contradicted ones automatically; it then chooses the next experiments itself to
  separate the survivors. So make them concrete, different from each other and falsifiable. Your own `actions` are used only when the code finds no experiment.
  `state` is read-only: state.cubes (list of cube ids); state.cell(id) -> (row, col) or None; state.height_of(id) -> stack index, 0 = on the floor;
  state.colour(id) -> ARC colour; state.stack(cell) -> cube ids bottom..top; state.height(cell) -> number of cubes on the cell; cells are tuples;
  state.tiles -> {{(row, col): colour}} of marked tiles as displayed in the board state (in imagined states: as displayed now);
  state.lit -> set of bordered cells as displayed (None in imagined states, so do not rely on it for predictions).
  state.heights() -> {{cell: number of cubes}} for every non-empty cell. Cells compare equal to lists or tuples.
  Only these builtins exist: len range all any sum min max abs sorted set dict list tuple enumerate zip isinstance int bool. No imports.
  Beware that all() of an empty sequence is True.
"""
HYPO_REPLY = """Reply with JSON only:
{{"observations": "<what changed and what it suggests>",
 "hypotheses": [{{"description": "<one sentence>", "code": "def lit(state):\\n    return False"}}],
 "hypothesis": "<one short Japanese sentence: your current best guess about the rule>", "notes": "<your notes>",
 "actions": [{{"name": "pick", "object_id": "...", "expect_change": false}}, {{"name": "place", "cell": [r, c], "expect_change": true}}]}}"""
SYSTEMS = {"plain": BASE + PLAIN_REPLY, "tricks": BASE + TRICKS_RULES + TRICKS_REPLY,
           "hypo": BASE + TRICKS_RULES + HYPO_RULES + HYPO_REPLY}


def _grid(g):
    return "\n".join(" ".join(str(v) for v in row) for row in g)


def stacks_of(obs):
    """{cell: [cube ids bottom..top]} rebuilt from an observation."""
    st = {}
    for c in sorted((c for c in obs["cubes"] if c["cell"] is not None), key=lambda c: c["height"]):
        st.setdefault(tuple(c["cell"]), []).append(c["id"])
    return st


def obs_text(obs, lessons, extra=""):
    cubes = "\n".join(f"- {c['id']}: colour={c['colour']} cell={c['cell']} height={c['height']}" for c in obs["cubes"])
    return (f"LEVEL {obs['level']}/{obs['n_levels']} | actions used in this level: {obs['actions']} | "
            f"lamp: {'ON' if obs['lamp'] else 'OFF'} | gripper holds: {obs['held'] or 'nothing'}\n"
            f"FLOOR (tile colours as displayed):\n{_grid(obs['floor'])}\n"
            f"TOP VIEW (cubes drawn over the tiles):\n{_grid(obs['top'])}\n"
            f"BORDERED (lit) CELLS: {obs['lit']}\nMARKED (non-black) CELLS NOT YET TRIED WITH ANY CUBE: {obs['untried']}\nCUBES:\n{cubes}\n"
            f"EXPERIMENT MEMORY (this level, oldest first; action -> visible effect):\n"
            + ("\n".join(obs["history"]) or "(no action yet)") +
            f"\nLESSONS FROM PREVIOUS LEVELS: {lessons or '(none)'}" + extra)


class AgentError(RuntimeError):
    pass


class QuotaError(Exception):
    """Daily free-tier quota used up: stop the whole run instead of counting levels as failed."""


def text_of(a):
    return f"pick({a['object_id']})" if a["name"] == "pick" else f"place({a['cell']})"


class LLMAgent:
    """Gemini agent; `mode` is 'plain', 'tricks' or 'hypo' (see above). `chat` can replace the API (tests)."""

    def __init__(self, mode="plain", chat=None):
        self.mode, self.requests = mode, 0
        self.model = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
        self.label = f"{self.model} [llm-{mode}]"
        self.system = SYSTEMS[mode].format(n=N, h=4, palette=", ".join(f"{i} {ARC[i][0]}" for i in ARC))
        if chat is None:
            from openai import OpenAI
            self.client = OpenAI(api_key=os.environ["GEMINI_API_KEY"], base_url=GEMINI_URL, max_retries=0, timeout=60)
        self._chat_fn = chat or self._api
        self.level, self.memory, self.lessons, self.ja, self.note = None, "", "", "", ""
        self.asked = 0

    def reset(self):
        pass

    # ---- API -----------------------------------------------------------------------------
    def _api(self, messages):
        """One request; retries with exponential backoff on HTTP 429 (free-tier rate limit), 5xx and timeouts."""
        from openai import APIConnectionError, InternalServerError, RateLimitError   # timeouts are connection errors
        for attempt in range(6):
            try:
                self.requests += 1
                r = self.client.chat.completions.create(model=self.model, messages=messages,
                                                        response_format={"type": "json_object"})
                return r.choices[0].message.content or ""
            except (RateLimitError, InternalServerError, APIConnectionError) as e:    # 429 / 5xx / timeout
                if "per day" in str(e):
                    raise QuotaError("daily free-tier quota exhausted; retry after it resets or use another model") from e
                wait = min(10 * 2 ** attempt, 90)
                log({"agent": self.label, "error": type(e).__name__, "detail": str(e)[:300], "retry_in_s": wait})
                time.sleep(wait)
        raise AgentError("rate limited too many times")

    # ---- per-level / per-turn bookkeeping ---------------------------------------------------
    def _new_level(self, obs):
        self.level, self.lessons = obs["level"], self.memory      # memory of the old level becomes the lessons
        self.queue, self.explore, self.pool, self.discarded = [], [], [], []
        self.states, self.visited, self.noeffect, self.dead, self.warned = [], set(), set(), [], set()
        self.last, self.pending, self.n0, self.fb_tried = None, None, 0, set()

    def _observe(self, obs):
        from .hypo import State, key_of
        stacks, tiles = stacks_of(obs), {(r, c): v for r, row in enumerate(obs["floor"]) for c, v in enumerate(row) if v}
        if obs["held"] is None:
            self.states.append((State(stacks, tiles, lit=frozenset(map(tuple, obs["lit"]))), obs["lamp"]))
            self.visited.add(key_of(stacks))
        if self.pending and obs["last_effect"] == "no visible change":
            key, desc = self.pending
            self.noeffect.add(key)
            if desc and desc not in self.dead:
                self.dead.append(desc)
        self.pending = None

    def _unexpected(self, obs):
        if self.last is None or self.last[1] is None:
            return False
        return (obs["last_effect"] != "no visible change") != self.last[1]

    def _emit(self, action, expect, obs):
        from .hypo import key_of
        a = {k: v for k, v in action.items() if k != "expect_change"}
        key = (key_of(stacks_of(obs)), obs["held"], json.dumps(a, sort_keys=True))
        desc = None
        if a["name"] == "place":
            cell = tuple(a["cell"])
            desc = f"place {obs['held']} on a tile of colour {obs['floor'][cell[0]][cell[1]]} holding {len(stacks_of(obs).get(cell, []))} cubes"
        self.pending, self.last = (key, desc), (a, expect)
        return {"hypothesis": self.ja or "(考え中)", "action": a, "note": self.note}

    def _trick_check(self, a, obs):
        """Cheap guards: block exact repeats of no-effect actions; warn once about revisiting a board state."""
        from .hypo import key_of
        st = stacks_of(obs)
        if (key_of(st), obs["held"], json.dumps(a, sort_keys=True)) in self.noeffect:
            return "this exact action had no visible effect in this exact board state before"
        if a["name"] == "place" and obs["held"]:
            after = {c: list(s) for c, s in st.items()}
            after.setdefault(tuple(a["cell"]), []).append(obs["held"])
            k = (key_of(after), json.dumps(a, sort_keys=True))
            if key_of(after) in self.visited and k not in self.warned:
                self.warned.add(k)
                return "this would bring the board back to a state you already visited; repeat it only if you want to re-check something"
        return None

    # ---- main entry ---------------------------------------------------------------------------
    def act(self, obs, legal):
        if obs["level"] != self.level:
            self._new_level(obs)
        self._observe(obs)
        if self.queue and self._unexpected(obs):
            self.queue = []
        for _ in range(3):
            while self.queue:
                a, expect = self.queue.pop(0)
                if legal(a) is None:
                    return self._emit(a, expect, obs)
                self.queue = []
            if self.mode == "hypo" and self._plan(obs):
                continue
            self._ask(obs, legal)
            if self.mode == "hypo" and self._plan(obs):
                continue
            self.queue = list(self.explore) or self._fallback_actions(obs)
        raise AgentError("no executable action")

    # ---- hypo: local falsification + discriminating experiments ----------------------------------
    def _refresh(self):
        from .hypo import falsify
        for h in list(self.pool):
            why = falsify(h, self.states)
            if why:
                self.pool.remove(h)
                self.discarded.append(f"{h.desc} -- {why}")

    def _plan(self, obs):
        from .hypo import search
        if obs["held"]:
            return False
        self._refresh()
        if not self.pool:
            return False
        floor = obs["floor"]
        tiles = {(r, c): v for r, row in enumerate(floor) for c, v in enumerate(row) if v}
        stacks = stacks_of(obs)
        spare = next(((r, c) for r in reversed(range(N)) for c in reversed(range(N)) if (r, c) not in tiles and (r, c) not in stacks))
        found, dropped = search(self.pool, stacks, tiles, sorted(set(tiles) | set(stacks) | {spare}), self.visited)
        self.discarded += [f"{h.desc} -- {why}" for h, why in dropped]
        if found is None:
            return False
        kind, path = found
        self.queue = [(a, None) for cube, cell in path for a in (pick(cube), place(cell))]
        self.note = f"仮説 {self.n0}→{len(self.pool)} | {'検証' if kind == 'test' else '実行'}: {text_of(self.queue[0][0])}→{text_of(self.queue[1][0])}"
        return True

    # ---- asking the LLM -----------------------------------------------------------------------
    def _extra(self, obs):
        if self.mode == "plain":
            return ""
        out = (f"\nWHAT CHANGED AFTER YOUR LAST ACTION: {obs['diff'] or '(nothing yet)'}"
               f"\nPLACEMENTS ALREADY TRIED WITHOUT ANY VISIBLE EFFECT: {self.dead or 'none'}"
               f"\nYOUR NOTES: {self.memory or '(none)'}")
        if self.mode == "hypo":
            self._refresh()
            out += "\nSURVIVING HYPOTHESES (consistent with everything observed, but code found no experiment to separate them):\n" + \
                   ("\n".join(f"- {h.desc}\n{h.code}" for h in self.pool) or "(none)")
            out += "\nOBSERVED BOARD STATES (gripper empty; heights by cell | lit cells | lamp):\n" + self._state_table()
            out += "\nDISCARDED HYPOTHESES (contradicted by observation):\n" + ("\n".join(self.discarded[-4:]) or "(none)")
        return out

    def _state_table(self):
        rows, seen = [], set()
        for st, lamp in self.states:
            line = f"{dict(st.heights())} | {sorted(map(list, st.lit or []))} | {'ON' if lamp else 'off'}"
            if line not in seen:
                seen.add(line)
                rows.append(line)
        return "\n".join(rows[-12:])

    def _parse(self, d, obs, strict):
        """Validate one answer; returns (actions as [(action, expect)], error). `strict` (first try of a request):
        also bounce answers with no action or with hypotheses that are not code; later tries accept them."""
        from .hypo import Hypothesis, falsify
        if not isinstance(d.get("hypothesis"), str):
            return None, "missing 'hypothesis' string"
        raw = [d.get("action")] if self.mode == "plain" else (d.get("actions") or [d.get("action")])
        raw = [a for a in raw if a][:4] if isinstance(raw, list) else []
        if not raw and strict and not (self.mode == "hypo" and self.pool):    # with survivors the planner can still act
            return None, "no action given: return at least one action"
        st, held, src = {c: list(s) for c, s in stacks_of(obs).items()}, obs["held"], None
        for i, a in enumerate(raw):
            err = check(st, held, a)
            if err:
                return None, f"action {i + 1}: {err}"
            if a["name"] == "pick":
                held = a["object_id"]
                src = next(c for c, s in st.items() if held in s)
                st[src].pop()
            else:
                if tuple(a["cell"]) == src:
                    return None, f"action {i + 1}: {held} is already on that cell; this move changes nothing"
                st.setdefault(tuple(a["cell"]), []).append(held)
                held = None
        err = self._trick_check(raw[0], obs) if raw and self.mode != "plain" else None
        if err:
            return None, err
        if self.mode == "hypo":
            new, errs = [], []
            for item in d.get("hypotheses") or []:
                try:
                    h = Hypothesis(item["description"], item["code"])
                except Exception as e:
                    errs.append(f"hypothesis must be an object with 'description' and Python 'code' defining lit(state) ({type(e).__name__}: {e})")
                    continue
                why = falsify(h, self.states)
                (errs.append(f"'{h.desc}' contradicts an observation: {why}") if why else new.append(h))
            if errs and strict:
                return None, "; ".join(errs[:2])
            if not new and not self.pool:
                return None, "no usable hypothesis: " + "; ".join(errs or ["none given"])
            have = {"".join(h.code.split()) for h in self.pool}          # drop repeats of hypotheses already held
            self.pool += [h for h in new if "".join(h.code.split()) not in have]
            self.n0 = len(self.pool)
        return [(dict(a), a.get("expect_change") if self.mode != "plain" else None) for a in raw], None

    def _fallback_actions(self, obs):
        """No action from the model or the planner: try a cube on a marked cell not tried yet. Each (cube, cell)
        is used at most once per level; when nothing is left the agent gives up instead of looping."""
        st = stacks_of(obs)
        tops = [s[-1] for s in st.values()]
        cells = [tuple(c) for c in obs["untried"]] or [(r, c) for r in range(N) for c in range(N)]
        if obs["held"]:
            return [(place(next(iter(cells))), None)]
        for cell in cells:
            for cube in tops:
                if (cube, cell) not in self.fb_tried and cell not in [c for c, s in st.items() if cube in s]:
                    self.fb_tried.add((cube, cell))
                    return [(pick(cube), None), (place(cell), None)]
        raise AgentError("no new experiment left")

    def _ask(self, obs, legal):
        self.asked += 1
        messages = [{"role": "system", "content": self.system},
                    {"role": "user", "content": obs_text(obs, self.lessons, self._extra(obs))}]
        for attempt in range(3):
            text = self._chat_fn(messages)
            try:
                d, _ = json.JSONDecoder().raw_decode(text[text.index("{"):])
                acts, err = self._parse(d, obs, attempt == 0) if isinstance(d, dict) else (None, "reply must be a JSON object")
            except (ValueError, KeyError, TypeError, AttributeError) as e:
                acts, err = None, f"not valid JSON or wrong shape ({type(e).__name__}: {e})"
            log({"agent": self.label, "level": obs["level"], "actions": obs["actions"],
                 "request": messages[-1]["content"], "response": text, "error": err})
            if err is None:
                self.ja = d["hypothesis"]
                self.memory = str(d.get("lesson" if self.mode == "plain" else "notes", self.memory))
                self.note = f"仮説 {self.n0}→{len(self.pool)}" if self.mode == "hypo" else ""
                self.explore = acts
                self.queue = [] if self.mode == "hypo" else list(acts)
                return
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": f"Invalid: {err}. Reply with corrected JSON only."}]
        raise AgentError("LLM could not produce a legal answer")


class ReplayAgent:
    """Replays the executed decisions of an earlier live run from its agent_log.jsonl (no API calls), e.g. to
    re-render the video. run.py logs every executed decision as {"agent", "decision"}; nothing new is decided."""

    def __init__(self, path):
        recs = [r for r in map(json.loads, open(path)) if "decision" in r]
        self.label = recs[0]["agent"]
        self.answers = iter(r["decision"] for r in recs)

    def reset(self):
        pass

    def act(self, obs, legal):
        return next(self.answers)


def load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass


def make_agent(name="llm-hypo", offline=False):
    """llm-plain / llm-tricks / llm-hypo if GEMINI_API_KEY is set (and not offline), else the offline heuristic."""
    load_env()
    if offline or os.getenv("PAI_OFFLINE") or not os.getenv("GEMINI_API_KEY"):
        return HeuristicAgent()
    return LLMAgent(name.removeprefix("llm-"))
