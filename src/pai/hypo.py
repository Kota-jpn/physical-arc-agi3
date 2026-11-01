"""Hypothesis programs for the llm-hypo agent.

The LLM writes competing hypotheses as tiny Python functions `lit(state) -> bool` ("is the lamp ON in this
board state?"). Plain code then (1) falsifies them against every state observed so far and (2) picks the next
experiment: the reachable board state on which the surviving hypotheses disagree most, or, if they all agree
that some reachable state lights the lamp, the shortest way there. Hypotheses may also use the cube weights the
arm has felt (state.mass); when an answer hinges on a cube not weighed yet, lifting that cube is the experiment.

BOUNDARY (no cheating): the game rules (game.feedback) are never called here. `apply_move` is only a
transition simulator for where cubes end up physically (a pick+place); whether the lamp is ON in a state we
have not visited is known only through the hypotheses, never through the true rules.
"""
import builtins
import re
import signal
from contextlib import contextmanager

from .game import CUBES, MAX_H
from .grid import N

SAFE = {n: getattr(builtins, n) for n in
        "len range all any sum min max abs sorted set frozenset dict list tuple enumerate zip isinstance int bool "
        "float str round map filter reversed".split()}


class Cell(tuple):
    """(row, col) that also equals the list [row, col]: models often write either."""
    def __eq__(self, other):
        return tuple(self) == tuple(other) if isinstance(other, (list, tuple)) else False

    __hash__ = tuple.__hash__


class Stack(list):
    """Cube ids bottom..top; equals a list or a tuple with the same ids."""
    def __eq__(self, other):
        return list(self) == list(other) if isinstance(other, (list, tuple)) else False

    __hash__ = None


class Kg(float):
    """A felt weight. Measurements are noisy and 0.1 + 0.2 != 0.3 in floats, so == means "within the resolution"."""
    RES = 0.015

    def __eq__(self, other):
        return abs(self - other) <= Kg.RES if isinstance(other, (int, float)) else False

    def __ne__(self, other):
        return not self == other

    __hash__ = float.__hash__

    def __add__(self, other):
        return Kg(float(self) + other)

    __radd__ = __add__

    def __sub__(self, other):
        return Kg(float(self) - other)

    def __rsub__(self, other):
        return Kg(other - float(self))


class Tiles(dict):
    """{(row, col): colour}; lookups also accept a list [row, col] or None (not a tile)."""
    def _k(self, key):
        return tuple(key) if isinstance(key, list) else key

    def get(self, key, default=None):
        return super().get(self._k(key), default)

    def __getitem__(self, key):
        return super().__getitem__(self._k(key))

    def __contains__(self, key):
        return super().__contains__(self._k(key))


class Unweighed(Exception):
    """A hypothesis asked for the mass of a cube the arm has not lifted yet."""


class State:
    """Read-only board state handed to hypothesis programs (see the prompt for the documented API)."""

    def __init__(self, stacks, tiles, lit=None, felt=None):
        self._stacks, self.tiles, self.lit = stacks, Tiles(tiles), lit
        self._felt = {} if felt is None else felt       # shared dict: masses are constant, so later weighings count too
        self.cubes = sorted(CUBES)

    def stack(self, cell):
        return Stack(self._stacks.get(tuple(cell), []))

    def height(self, cell):
        return len(self._stacks.get(tuple(cell), []))

    def cell(self, cube):
        c = next((c for c, s in self._stacks.items() if cube in s), None)
        return None if c is None else Cell(c)

    def height_of(self, cube):
        c = self.cell(cube)
        return None if c is None else self._stacks[c].index(cube)

    def heights(self):
        """{cell: number of cubes} for every cell that holds at least one cube."""
        return {Cell(c): len(s) for c, s in self._stacks.items() if s}

    def colour(self, cube):
        return CUBES[cube]

    def mass(self, cube):
        if cube not in self._felt:
            raise Unweighed(cube)
        return Kg(self._felt[cube])


class Timeout(Exception):
    pass


@contextmanager
def limit(seconds):
    def handler(*_):
        raise Timeout()
    old = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


class Hypothesis:
    def __init__(self, desc, code):
        self.desc, self.code = str(desc), str(code)
        ns = {}
        with limit(1):
            exec(self.code, {"__builtins__": SAFE}, ns)   # restricted builtins: no imports, no file access
        self.fn = ns.get("lit")
        if not callable(self.fn):
            raise ValueError("code must define lit(state)")


def key_of(stacks):
    return tuple(sorted((c, tuple(s)) for c, s in stacks.items() if s))


def falsify(h, history):
    """None if `h` agrees with every (state, lamp) observed so far, otherwise the reason it is discarded."""
    try:
        with limit(2):
            for state, lamp in history:
                try:
                    pred = bool(h.fn(state))
                except Unweighed:                       # cannot be checked until that cube is weighed
                    continue
                if pred != lamp:
                    return (f"predicted lamp {'ON' if not lamp else 'OFF'} but it was {'ON' if lamp else 'OFF'} with cubes {key_of(state._stacks)}"
                            + (" (check for conditions that are vacuously true, e.g. all() over an empty list)" if not lamp else ""))
    except Timeout:
        return "timed out"
    except Exception as e:
        return f"crashed: {type(e).__name__}: {e}"
    return None


def apply_move(stacks, cube, cell):
    """Physical transition of one pick+place: where the cubes end up. Returns None if impossible."""
    src = next(c for c, s in stacks.items() if cube in s)
    if stacks[src][-1] != cube or src == cell or len(stacks.get(cell, [])) >= MAX_H:
        return None
    new = {c: list(s) for c, s in stacks.items()}
    new[src].pop()
    new.setdefault(cell, []).append(cube)
    return {c: s for c, s in new.items() if s}


def search(hyps, stacks, tiles, cells, visited, max_depth=6, felt=None):
    """Breadth-first over pick+place moves. Returns (kind, [(cube, cell), ...]) or None, plus discarded hyps.

    kind 'plan': every surviving hypothesis that answers predicts the lamp ON in the final state.
    kind 'test': the surviving hypotheses disagree on the final state (largest split wins).
    kind 'weigh': no test or plan is decidable with the weights known so far; lift the cube whose unknown
    weight the hypotheses asked about most, onto the cell where most of them could not answer."""
    seen, frontier, dropped = set(visited) | {key_of(stacks)}, [(stacks, [])], []
    asked = {}                                          # (cube, first move) -> hypotheses that needed its weight
    for _ in range(max_depth):
        new = []
        for st, path in frontier:
            for cube in CUBES:
                for cell in cells:
                    nxt = apply_move(st, cube, cell)
                    if nxt is None or key_of(nxt) in seen:
                        continue
                    seen.add(key_of(nxt))
                    new.append((nxt, path + [(cube, cell)]))
        preds = []
        for h in list(hyps):
            try:
                with limit(3):
                    preds.append([_pred(h, st, tiles, felt, asked, path[0]) for st, path in new])
            except Timeout:
                dropped.append((h, "timed out"))
                hyps.remove(h)
        best, best_score = None, (0, 0)
        for i, (st, path) in enumerate(new):
            votes = [p[i] for p in preds if p[i] is not None]
            unsure = any(p[i] is UNWEIGHED for p in preds)
            votes = [v for v in votes if v is not UNWEIGHED]
            if votes and all(votes) and not unsure:
                return ("plan", path), dropped
            score = (min(votes.count(True), votes.count(False)), votes.count(True))   # larger split, then more 'ON' votes
            if score[0] > 0 and score > best_score:
                best, best_score = path, score
        if best:
            return ("test", best), dropped
        frontier = new
    if asked:
        cube = max(CUBES, key=lambda c: sum(n for (k, _), n in asked.items() if k == c))
        move = max((m for k, m in asked if k == cube), key=lambda m: asked[(cube, m)])
        return ("weigh", [move]), dropped
    return None, dropped


UNWEIGHED = "unweighed"     # prediction marker: the answer depends on a weight not known yet


def cells_in(code):
    """Board cells written as literals in a hypothesis, e.g. (2, 1) or [2, 1]."""
    return {(int(r), int(c)) for r, c in re.findall(r"[\(\[]\s*(\d)\s*,\s*(\d)\s*[\)\]]", code) if int(r) < N and int(c) < N}


def ever_on(h, stacks, tiles, cells, felt=None, max_depth=6):
    """False if `h` predicts the lamp OFF in every state reachable within `max_depth` moves: such a hypothesis
    can be neither tested nor used to solve the level. True if some state may light it (or it cannot tell)."""
    seen, frontier = {key_of(stacks)}, [stacks]
    try:
        with limit(3):
            for _ in range(max_depth):
                new = []
                for st in frontier:
                    for cube in CUBES:
                        for cell in cells:
                            nxt = apply_move(st, cube, cell)
                            if nxt is None or key_of(nxt) in seen:
                                continue
                            seen.add(key_of(nxt))
                            if _pred(h, nxt, tiles, felt) in (True, UNWEIGHED):
                                return True
                            new.append(nxt)
                frontier = new
    except Timeout:
        return True
    return False


def _pred(h, stacks, tiles, felt=None, asked=None, first=None):
    """Prediction on a hypothetical state; None if the program cannot answer (e.g. it reads state.lit),
    UNWEIGHED if it needs the weight of a cube not lifted yet (counted in `asked` per first move)."""
    try:
        return bool(h.fn(State(stacks, tiles, lit=None, felt=felt)))
    except Timeout:
        raise
    except Unweighed as e:
        cube = e.args[0]
        if asked is not None and cube in CUBES and first is not None and first[0] == cube:
            asked[(cube, first)] = asked.get((cube, first), 0) + 1
        return UNWEIGHED
    except Exception:
        return None
