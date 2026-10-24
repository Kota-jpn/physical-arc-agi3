"""Game logic of "Physical ARC-AGI-3" (unofficial, inspired by ARC-AGI-3). Pure Python, no physics.

The robot only executes actions. The state of the game is *derived* from observed cube positions
(`grid.stacks_from_poses`): {cell: [cube ids bottom..top]}. This module decides the visual feedback
(tile colours, bordered/"lit" tiles, lamp) from that state. The rules are hidden from the agent.
"""
from dataclasses import dataclass, field

from .grid import N

CUBES = {"red": 2, "blue": 1, "green": 3, "yellow": 4}     # cube id -> ARC colour
MAX_H = 4                                                    # max stack height
GREY = 5


@dataclass
class Level:
    n: int
    kind: str                       # "match" | "switch" | "count"
    tiles: dict                     # cell -> colour shown on the floor (everything else is black 0)
    start: dict                     # cube id -> start cell
    hidden: dict = field(default_factory=dict)   # switch level: grey cell -> colour revealed by the switch
    switch: tuple = None            # switch level: the switch tile
    target: tuple = None            # count level: the tile to build the tower on
    markers: tuple = ()             # count level: the N marker squares
    optimal: int = 6                # fewest actions that solve it


LEVELS = [
    # L1: a cube on the tile of its own colour lights the tile; lamp when every marked tile is lit
    Level(1, "match", tiles={(1, 1): 2, (2, 4): 1, (4, 2): 3},
          start={"red": (4, 5), "blue": (0, 3), "green": (3, 0), "yellow": (5, 3)}),
    # L2: targets stay grey until any cube rests on the orange switch tile; then colours show
    Level(2, "switch", tiles={(3, 1): 7, (1, 4): GREY, (4, 4): GREY},
          start={"red": (0, 2), "blue": (5, 2), "green": (2, 0), "yellow": (2, 5)},
          switch=(3, 1), hidden={(1, 4): 2, (4, 4): 1}),
    # L3: N markers shown; a tower of exactly N cubes on the light-blue tile lights the lamp
    Level(3, "count", tiles={(0, 0): 6, (0, 1): 6, (0, 2): 6, (3, 3): 8},
          start={"red": (2, 5), "blue": (5, 1), "green": (4, 5), "yellow": (1, 0)},
          target=(3, 3), markers=((0, 0), (0, 1), (0, 2))),
]


@dataclass
class Feedback:
    floor: dict      # cell -> colour as currently displayed (marked cells only)
    lit: frozenset   # cells drawn bright / bordered
    lamp: bool       # level-complete lamp


def feedback(level, stacks):
    """The hidden rules. stacks: {cell: [cube ids bottom..top]} (held cube excluded)."""
    height = lambda cell: len(stacks.get(cell, []))
    bottom_colour = lambda cell: CUBES[stacks[cell][0]] if stacks.get(cell) else None
    floor, lit = dict(level.tiles), set()
    if level.kind == "match":
        lit = {c for c, col in level.tiles.items() if bottom_colour(c) == col}
        lamp = lit == set(level.tiles)
    elif level.kind == "switch":
        pressed = height(level.switch) > 0
        if pressed:
            floor.update(level.hidden)
            lit.add(level.switch)
            lit |= {c for c, col in level.hidden.items() if bottom_colour(c) == col}
        lamp = pressed and all(c in lit for c in level.hidden)
    else:
        h = height(level.target)
        lit = {m for k, m in enumerate(level.markers, 1) if h >= k}
        lamp = h == len(level.markers)
    return Feedback(floor, frozenset(lit), lamp)


def describe_effect(before, after):
    """Words for what visibly changed between two feedbacks (goes into the agent's history)."""
    parts = []
    changed = sorted(c for c in after.floor if after.floor[c] != before.floor.get(c))
    if changed:
        parts.append("floor tiles changed colour: " +
                     ", ".join(f"[{r},{c}] {before.floor[(r, c)]}->{after.floor[(r, c)]}" for r, c in changed))
    if after.lit - before.lit:
        parts.append("newly bordered/lit: " + ", ".join(f"[{r},{c}]" for r, c in sorted(after.lit - before.lit)))
    if before.lit - after.lit:
        parts.append("no longer lit: " + ", ".join(f"[{r},{c}]" for r, c in sorted(before.lit - after.lit)))
    if after.lamp != before.lamp:
        parts.append("lamp turned " + ("ON" if after.lamp else "OFF"))
    return "; ".join(parts) or "no visible change"


def check(stacks, held, action):
    """None if `action` is executable on this layout ({cell: [cubes]}, held cube), else the reason."""
    if not isinstance(action, dict):
        return "action must be a JSON object"
    name = action.get("name")
    if name == "pick":
        cube = action.get("object_id")
        if cube not in CUBES:
            return f"unknown object_id {cube!r}; cubes are {sorted(CUBES)}"
        if held:
            return f"the gripper already holds {held}; place it first"
        stack = next(s for s in stacks.values() if cube in s)
        if stack[-1] != cube:
            return f"{cube} has {stack[-1]} on top of it; only the top cube can be picked"
        return None
    if name == "place":
        cell = action.get("cell")
        if not (isinstance(cell, (list, tuple)) and len(cell) == 2 and all(isinstance(v, int) for v in cell)
                and 0 <= cell[0] < N and 0 <= cell[1] < N):
            return f"cell must be [row, col] with 0 <= row, col < {N}"
        if not held:
            return "the gripper is empty; pick a cube first"
        if len(stacks.get(tuple(cell), [])) >= MAX_H:
            return f"cell {list(cell)} already holds {MAX_H} cubes"
        return None
    return "action name must be 'pick' or 'place'"


class Game:
    def __init__(self, level=1):
        self.total = 0
        self.start_level(level)

    def start_level(self, n):
        self.level = LEVELS[n - 1]
        self.stacks = {}
        for cube, cell in self.level.start.items():
            self.stacks.setdefault(cell, []).append(cube)
        self.held, self.actions = None, 0
        self.fb = feedback(self.level, self.stacks)
        self.last_effect, self.last_diff = "", ""
        self.history, self.tried = [], set()        # experiment memory of this level; cells a cube was placed on

    @property
    def solved(self):
        return self.fb.lamp

    def cell_of(self, cube):
        return next((c for c, s in self.stacks.items() if cube in s), None)

    # ---- actions -----------------------------------------------------------
    def legal(self, action):
        """None if the action may be executed, otherwise the reason it cannot."""
        return check(self.stacks, self.held, action)

    def step(self, action, observed=None):
        """Execute an action and count it. The new cube layout is applied symbolically, unless
        `observed=(stacks, held)` is given (derived from the physics simulation)."""
        assert self.legal(action) is None, self.legal(action)
        before, heights, top0 = self.fb, {c: len(s) for c, s in self.stacks.items()}, self.top_grid()
        if observed is not None:
            self.stacks, self.held = observed
        elif action["name"] == "pick":
            self.held = action["object_id"]
            self.stacks[self.cell_of(self.held)].pop()
        else:
            self.stacks.setdefault(tuple(action["cell"]), []).append(self.held)
            self.held = None
        self.stacks = {c: s for c, s in self.stacks.items() if s}
        self.fb = feedback(self.level, self.stacks)
        self.actions += 1
        self.total += 1
        effect = describe_effect(before, self.fb)
        if action["name"] == "pick" and self.held != action["object_id"]:
            effect = "the grasp failed (nothing is held); " + effect
        text = f"pick({action['object_id']})" if action["name"] == "pick" else f"place({list(action['cell'])})"
        if action["name"] == "place":
            self.tried.add(tuple(action["cell"]))
        top1 = self.top_grid()
        cells = [f"[{r},{c}] {top0[r][c]}->{top1[r][c]}" for r in range(N) for c in range(N) if top0[r][c] != top1[r][c]]
        self.last_effect = effect
        self.last_diff = "top-view cells changed: " + (", ".join(cells) or "none") + "; " + effect
        now = {c: len(s) for c, s in self.stacks.items()}
        moved = ", ".join(f"[{r},{c}] {heights.get((r, c), 0)}->{now.get((r, c), 0)}"
                          for r, c in sorted(set(heights) | set(now)) if heights.get((r, c), 0) != now.get((r, c), 0))
        self.history.append(f"#{self.actions} {text} -> {effect}" + (f" (stack heights: {moved})" if moved else ""))
        return effect

    # ---- what the agent sees ---------------------------------------------------
    def top_grid(self):
        """ARC-style integer grid: cube colour where a cube lies, otherwise the tile colour."""
        g = [[self.fb.floor.get((r, c), 0) for c in range(N)] for r in range(N)]
        for (r, c), s in self.stacks.items():
            g[r][c] = CUBES[s[-1]]
        return g

    def observation(self):
        floor = [[self.fb.floor.get((r, c), 0) for c in range(N)] for r in range(N)]
        cubes = []
        for cube, col in CUBES.items():
            cell = self.cell_of(cube)
            cubes.append({"id": cube, "colour": col, "cell": list(cell) if cell else None,
                          "height": self.stacks[cell].index(cube) if cell else None})
        return {"level": self.level.n, "n_levels": len(LEVELS), "actions": self.actions,
                "lamp": self.fb.lamp, "held": self.held, "floor": floor, "top": self.top_grid(),
                "lit": sorted(list(c) for c in self.fb.lit), "cubes": cubes,
                "history": self.history[-40:], "last_effect": self.last_effect, "diff": self.last_diff,
                "untried": sorted(list(c) for c in self.fb.floor if c not in self.tried)}
