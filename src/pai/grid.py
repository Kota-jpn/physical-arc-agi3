"""Board geometry (cell <-> world) and the ARC colour palette. No game rules here."""

N = 6                      # board is N x N cells; cell = (row, col); row 0 is nearest to the robot
PITCH = 0.07               # cell pitch [m]
X0, Y0 = 0.31, -0.21       # world xy of the board corner (min x, min y)
CUBE = 0.04                # cube edge [m]

# standard ARC palette: index -> (name, hex)
ARC = {0: ("black", "#000000"), 1: ("blue", "#0074D9"), 2: ("red", "#FF4136"),
       3: ("green", "#2ECC40"), 4: ("yellow", "#FFDC00"), 5: ("grey", "#AAAAAA"),
       6: ("magenta", "#F012BE"), 7: ("orange", "#FF851B"), 8: ("light-blue", "#7FDBFF"),
       9: ("maroon", "#870C25")}


def rgb(i):
    h = ARC[i][1]
    return tuple(int(h[k:k + 2], 16) for k in (1, 3, 5))


def cell_center(cell):
    r, c = cell
    return X0 + PITCH * (r + 0.5), Y0 + PITCH * (c + 0.5)


def snap(x, y):
    """Nearest board cell of a world position (clamped to the board)."""
    r = int((x - X0) // PITCH)
    c = int((y - Y0) // PITCH)
    return min(max(r, 0), N - 1), min(max(c, 0), N - 1)


def stacks_from_poses(poses):
    """{cube id: xyz} -> {cell: [cube ids bottom..top]} by snapping to cells and sorting by height."""
    stacks = {}
    for name, p in sorted(poses.items(), key=lambda kv: kv[1][2]):
        stacks.setdefault(snap(p[0], p[1]), []).append(name)
    return stacks
