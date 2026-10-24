"""Hypothesis programs for the llm-hypo agent.

The LLM writes competing hypotheses as tiny Python functions `lit(state) -> bool` ("is the lamp ON in this
board state?"). Plain code then (1) falsifies them against every state observed so far and (2) picks the next
experiment: the reachable board state on which the surviving hypotheses disagree most, or, if they all agree
that some reachable state lights the lamp, the shortest way there.

BOUNDARY (no cheating): the game rules (game.feedback) are never called here. `apply_move` is only a
transition simulator for where cubes end up physically (a pick+place); whether the lamp is ON in a state we
have not visited is known only through the hypotheses, never through the true rules.
"""
import builtins
import signal
from contextlib import contextmanager

from .game import CUBES, MAX_H

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


class State:
    """Read-only board state handed to hypothesis programs (see the prompt for the documented API)."""

    def __init__(self, stacks, tiles, lit=None):
        self._stacks, self.tiles, self.lit = stacks, Tiles(tiles), lit
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
                if bool(h.fn(state)) != lamp:
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


def search(hyps, stacks, tiles, cells, visited, max_depth=3):
    """Breadth-first over pick+place moves. Returns (kind, [(cube, cell), ...]) or None, plus discarded hyps.

    kind 'plan': every surviving hypothesis that answers predicts the lamp ON in the final state.
    kind 'test': the surviving hypotheses disagree on the final state (largest split wins)."""
    seen, frontier, dropped = set(visited) | {key_of(stacks)}, [(stacks, [])], []
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
                    preds.append([_pred(h, st, tiles) for st, _ in new])
            except Timeout:
                dropped.append((h, "timed out"))
                hyps.remove(h)
        best, best_score = None, (0, 0)
        for i, (st, path) in enumerate(new):
            votes = [p[i] for p in preds if p[i] is not None]
            if votes and all(votes):
                return ("plan", path), dropped
            score = (min(votes.count(True), votes.count(False)), votes.count(True))   # larger split, then more 'ON' votes
            if score[0] > 0 and score > best_score:
                best, best_score = path, score
        if best:
            return ("test", best), dropped
        frontier = new
    return None, dropped


def _pred(h, stacks, tiles):
    """Prediction on a hypothetical state; None if the program cannot answer (e.g. it reads state.lit)."""
    try:
        return bool(h.fn(State(stacks, tiles, lit=None)))
    except Timeout:
        raise
    except Exception:
        return None
