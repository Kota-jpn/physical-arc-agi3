"""LLM task planner: natural language -> validated JSON plan of primitives.

Plan schema:  {"steps": [{"action": "pick", "object": "red_cube"},
                         {"action": "place_on", "target": "blue_plate"}, ...]}

Backends (auto-selected, see `get_backend`):
  1. Azure OpenAI   (AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY / AZURE_OPENAI_DEPLOYMENT_NAME)
  2. OpenAI         (OPENAI_API_KEY, OPENAI_MODEL)
  3. offline        (JSON cache of earlier LLM answers, then a tiny rule-based parser)
"""
import hashlib
import json
import os
import re
from pathlib import Path

from .scene import OBJECTS, DESCRIPTIONS

CACHE = Path(__file__).resolve().parents[2] / "plan_cache.json"

SYSTEM_PROMPT = """You are the task planner of a Franka Panda robot arm on a table.
The user gives an instruction in Japanese or English. Convert it into a JSON plan.

Scene objects (use these ids exactly):
{objects}

Primitives:
  {{"action": "pick", "object": <id of a cube>}}        grasp a cube with the gripper
  {{"action": "place_on", "target": <id>}}               put the HELD cube on top of the target
                                                       (target = another cube or a plate)
Rules:
- Every pick must be immediately followed by exactly one place_on.
- Only cubes can be picked. The plate cannot be moved.
- You cannot pick a cube that has another object on top of it, and you cannot place on
  a cube that already has something on it. Unstack first if needed.
- A plate holds at most one stack. Cubes stack one per level.
- Order matters: do what the user says in the order that makes the final state correct.
- If you are given an OBSERVED STATE and a failure report, the world has changed or a step
  failed. Output a NEW plan containing only the steps still needed from the current state.
Return ONLY a JSON object: {{"reason": "<one short sentence>", "steps": [...]}}"""


# ----------------------------------------------------------------------------- state
def describe_state(on, held=None):
    """Symbolic state text. `on` maps object -> support ('table' or an object id)."""
    lines = [f"- {k}: on {v}" for k, v in on.items() if OBJECTS[k][0] == "cube"]
    lines.append(f"- gripper holds: {held or 'nothing'}")
    return "\n".join(lines)


def initial_on():
    return {k: "table" for k, v in OBJECTS.items() if v[0] == "cube"}


# ----------------------------------------------------------------------------- validation
class PlanError(ValueError):
    pass


def validate(plan, on):
    """Symbolically execute `plan` from state `on`; raise PlanError if invalid."""
    if not isinstance(plan, dict) or not isinstance(plan.get("steps"), list) or not plan["steps"]:
        raise PlanError("plan must be {'steps': [..]} with at least one step")
    on = dict(on)
    held = None
    for i, s in enumerate(plan["steps"], 1):
        a = s.get("action")
        if a == "pick":
            o = s.get("object")
            if o not in OBJECTS or OBJECTS[o][0] != "cube":
                raise PlanError(f"step {i}: '{o}' is not a pickable cube")
            if held:
                raise PlanError(f"step {i}: gripper already holds {held}")
            if any(v == o for v in on.values()):
                raise PlanError(f"step {i}: {o} has something on top; unstack it first")
            held = o
            on[o] = "gripper"
        elif a == "place_on":
            t = s.get("target")
            if not held:
                raise PlanError(f"step {i}: place_on without a held cube")
            if t not in OBJECTS or t == held:
                raise PlanError(f"step {i}: invalid target '{t}'")
            if any(v == t for k, v in on.items() if k != held):
                raise PlanError(f"step {i}: {t} already has something on top")
            on[held] = t
            held = None
        else:
            raise PlanError(f"step {i}: unknown action '{a}'")
    if held:
        raise PlanError("plan ends while still holding a cube")
    return on


def pairs(plan):
    """[(object, target)] pairs of a validated plan."""
    out, cur = [], None
    for s in plan["steps"]:
        if s["action"] == "pick":
            cur = s["object"]
        else:
            out.append((cur, s["target"]))
    return out


# ----------------------------------------------------------------------------- offline parser
_COLORS = {"赤": "red", "あか": "red", "red": "red", "青": "blue", "あお": "blue", "blue": "blue",
           "緑": "green", "みどり": "green", "green": "green", "黄": "yellow", "黄色": "yellow",
           "yellow": "yellow"}
_REF = re.compile(r"(黄色|赤|青|緑|あか|あお|みどり|red|blue|green|yellow)(?:い|色|の)?\s*(キューブ|立方体|皿|さら|plate|cube)?",
                  re.I)


def _resolve(color, noun):
    color = _COLORS[color.lower()]
    if noun and noun.lower() in ("皿", "さら", "plate"):
        return f"{color}_plate"
    return f"{color}_cube"


def rule_based_plan(instruction):
    """Tiny parser: in each clause, first colour = object to move, second = target."""
    clauses = re.split(r"そのあと|その後|それから|次に|、|,|\bthen\b|\band then\b|。", instruction)
    steps = []
    for c in clauses:
        refs = [_resolve(m.group(1), m.group(2)) for m in _REF.finditer(c)]
        if len(refs) >= 2 and refs[0] in OBJECTS and refs[1] in OBJECTS:
            steps += [{"action": "pick", "object": refs[0]},
                      {"action": "place_on", "target": refs[1]}]
    return {"reason": "rule-based offline parser", "steps": steps}


def rule_based_replan(goal_pairs, on):
    """Offline re-plan: redo every unmet (object, target) goal in order."""
    steps = []
    for o, t in goal_pairs:
        if on.get(o) != t:
            steps += [{"action": "pick", "object": o}, {"action": "place_on", "target": t}]
    return {"reason": "rule-based offline re-plan: redo unmet goals", "steps": steps}


# ----------------------------------------------------------------------------- LLM backends
def _load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    except ImportError:
        pass


def get_backend():
    """Return (name, callable(messages)->str) or ('offline', None)."""
    _load_env()
    if os.getenv("PAI_OFFLINE"):
        return "offline", None
    if os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_API_KEY"):
        from openai import AzureOpenAI
        model = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5.1")
        client = AzureOpenAI(azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                             api_key=os.environ["AZURE_OPENAI_API_KEY"],
                             api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"))
    elif os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI
        model = os.getenv("OPENAI_MODEL", "gpt-5.1")
        client = OpenAI()
    else:
        return "offline", None

    def call(messages):
        r = client.chat.completions.create(model=model, messages=messages,
                                           response_format={"type": "json_object"})
        return r.choices[0].message.content
    return f"LLM:{model}", call


def _cache_get(key):
    if CACHE.exists():
        return json.loads(CACHE.read_text()).get(key)


def _cache_put(key, plan):
    d = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    d[key] = plan
    CACHE.write_text(json.dumps(d, ensure_ascii=False, indent=1))


def _extract_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0))


class Planner:
    def __init__(self):
        self.backend, self._call = get_backend()
        objs = "\n".join(f"- {k}: {DESCRIPTIONS[k]}" for k in OBJECTS)
        self.system = SYSTEM_PROMPT.format(objects=objs)
        self.log = []   # (kind, text) for the HUD / README

    def _ask(self, user_text, on, offline_fallback, max_tries=3):
        """Ask the LLM, validate, and feed validation errors back (self-repair)."""
        key = hashlib.sha1((self.system + user_text).encode()).hexdigest()[:16]
        if self._call is None:
            plan = _cache_get(key)
            src = "cache"
            if plan is None:
                plan, src = offline_fallback(), "rule-based"
            validate(plan, on)
            self.backend_used = f"offline ({src})"
            return plan
        messages = [{"role": "system", "content": self.system},
                    {"role": "user", "content": user_text}]
        for _ in range(max_tries):
            text = self._call(messages)
            try:
                plan = _extract_json(text)
                validate(plan, on)
                _cache_put(key, plan)
                self.backend_used = self.backend
                return plan
            except (PlanError, ValueError, KeyError) as e:
                messages += [{"role": "assistant", "content": text},
                             {"role": "user", "content": f"Invalid plan: {e}. Fix it and return JSON only."}]
                self.log.append(("validation-error", str(e)))
        raise PlanError("LLM could not produce a valid plan")

    def plan(self, instruction, on):
        text = f"INSTRUCTION: {instruction}\nOBSERVED STATE:\n{describe_state(on)}"
        return self._ask(text, on, lambda: rule_based_plan(instruction))

    def replan(self, instruction, original, on, held, failure):
        text = (f"INSTRUCTION: {instruction}\nORIGINAL PLAN: {json.dumps(original['steps'])}\n"
                f"FAILURE: {failure}\nOBSERVED STATE:\n{describe_state(on, held)}\n"
                "Output a new plan for the remaining work.")
        return self._ask(text, on, lambda: rule_based_replan(pairs(original), on))
