"""Genesis tabletop scene for Physical ARC-AGI-3: board tiles, lamp, cubes + Franka Panda controller.

The floor tiles / lamp are purely visual (no collision) and are only *displayed* here: what they
show is decided by the rules in game.py. Fixed tile entities are moved with set_pos to change colour.
"""
import numpy as np
import genesis as gs

from .game import CUBES
from .grid import ARC, CUBE, N, PITCH, X0, Y0, cell_center, rgb

TILE = 0.058                    # tile edge; the frame below is PITCH wide
POOL = 6                        # coloured tile entities available per colour
HIDDEN = -0.2                   # z where unused visual entities wait
LAMP_XY = (X0 + PITCH * N / 2, Y0 + PITCH * N + 0.07)


def col(i):
    return tuple(v / 255 for v in rgb(i))


OPEN = 0.03           # finger opening (per finger): narrow, so neighbouring cells are not touched
HAND_OFFSET = 0.105   # hand-link height above cube centre when grasping
QUAT_DOWN = np.array([0.0, 1.0, 0.0, 0.0])


class Sim:
    def __init__(self, res=(1280, 720), render_every=4, hook=None):
        gs.init(backend=gs.cpu, logging_level="error")
        self.scene = gs.Scene(
            show_viewer=False,
            sim_options=gs.options.SimOptions(dt=0.01, substeps=2),
            vis_options=gs.options.VisOptions(show_world_frame=False),
        )
        self.scene.add_entity(gs.morphs.Plane(), surface=gs.surfaces.Default(color=(0.02, 0.02, 0.02)))
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
        self._build_board()
        self.ent = {}
        for name, colour in CUBES.items():
            self.ent[name] = self.scene.add_entity(
                gs.morphs.Box(size=(CUBE,) * 3, pos=(0, 0, HIDDEN)),
                material=gs.materials.Rigid(friction=2.0), surface=gs.surfaces.Default(color=col(colour)))
        self.cam = self.scene.add_camera(res=res, pos=(1.0, -0.72, 0.85), lookat=(0.52, 0.0, 0.03),
                                         fov=45, GUI=False)
        self.scene.build()
        self.hand = self.robot.get_link("hand")
        self.arm = np.arange(7)
        self.fingers = np.array([7, 8])
        self.robot.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
        self.robot.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
        self.robot.set_dofs_force_range(
            np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
            np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]))
        self.render_every, self.hook, self.n = render_every, hook, 0
        home = np.array([0, -0.6, 0, -2.4, 0, 1.8, 0.785, 0.04, 0.04])
        self.robot.set_qpos(home)
        self.home = home
        self.robot.control_dofs_position(home)
        for _ in range(30):
            self.step()

    # ---- board (visual only) ---------------------------------------------
    def _visual(self, morph_cls, colour, **kw):
        return self.scene.add_entity(morph_cls(collision=False, fixed=True, **kw),
                                     surface=gs.surfaces.Default(color=colour))

    def _build_board(self):
        side = PITCH * N + 0.02
        self._visual(gs.morphs.Box, (0.2, 0.2, 0.2), size=(side, side, 0.002),
                     pos=(X0 + PITCH * N / 2, Y0 + PITCH * N / 2, 0.001))
        self.frame, self.pool = {}, {c: [] for c in range(1, 10)}
        for r in range(N):
            for c in range(N):
                x, y = cell_center((r, c))
                self.frame[(r, c)] = self._visual(gs.morphs.Box, (1, 1, 1), size=(PITCH - 0.002,) * 2 + (0.003,),
                                                  pos=(x, y, HIDDEN))
                self._visual(gs.morphs.Box, col(0), size=(TILE, TILE, 0.004), pos=(x, y, 0.002))
        for colour, lst in self.pool.items():
            lst += [self._visual(gs.morphs.Box, col(colour), size=(TILE, TILE, 0.0045), pos=(0, 0, HIDDEN))
                    for _ in range(POOL)]
        self.shown = {}                                  # cell -> (colour, entity) currently on the floor
        lx, ly = LAMP_XY
        self._visual(gs.morphs.Box, (0.25, 0.25, 0.25), size=(0.05, 0.05, 0.08), pos=(lx, ly, 0.04))
        self.bulb = {on: self._visual(gs.morphs.Sphere, col(3 if on else 5), radius=0.04, pos=(lx, ly, HIDDEN if on else 0.12))
                     for on in (False, True)}

    def show(self, fb):
        """Display a game.Feedback: tile colours, bordered (lit) tiles and the lamp."""
        want = {c: v for c, v in fb.floor.items() if v}
        for cell, (colour, e) in list(self.shown.items()):
            if want.get(cell) != colour:
                e.set_pos(np.array([0, 0, HIDDEN]))
                self.pool[colour].append(e)
                del self.shown[cell]
        for cell, colour in want.items():
            if cell not in self.shown:
                e = self.pool[colour].pop()
                x, y = cell_center(cell)
                e.set_pos(np.array([x, y, 0.00225]))
                self.shown[cell] = (colour, e)
        for cell, e in self.frame.items():
            x, y = cell_center(cell)
            e.set_pos(np.array([x, y, 0.0015 if cell in fb.lit else HIDDEN]))
        lx, ly = LAMP_XY
        self.bulb[fb.lamp].set_pos(np.array([lx, ly, 0.12]))
        self.bulb[not fb.lamp].set_pos(np.array([lx, ly, HIDDEN]))

    def load_level(self, level, fb):
        """Put the cubes of `level` on their start cells (all cubes exist in every level) and show `fb`."""
        self.show(fb)
        for name, cell in level.start.items():
            x, y = cell_center(cell)
            self.ent[name].set_pos(np.array([x, y, CUBE / 2]))
            self.ent[name].set_quat(np.array([1.0, 0, 0, 0]))
        self.open_gripper()
        self.hold(30)

    # ---- stepping / recording hook -------------------------------------
    def step(self):
        self.scene.step()
        self.n += 1
        if self.hook and self.n % self.render_every == 0:
            rgb, *_ = self.cam.render()
            self.hook(rgb)

    def hold(self, steps):
        for _ in range(steps):
            self.step()

    # ---- state ----------------------------------------------------------
    def pos(self, name):
        return self.ent[name].get_pos().cpu().numpy()

    def poses(self):
        return {k: self.pos(k) for k in self.ent}

    def hand_pos(self):
        return self.hand.get_pos().cpu().numpy()

    # ---- motion ---------------------------------------------------------
    def move_to(self, xyz, steps=120):
        q_goal = self.robot.inverse_kinematics(link=self.hand, pos=np.asarray(xyz), quat=QUAT_DOWN)
        q_goal = q_goal.cpu().numpy() if hasattr(q_goal, "cpu") else np.asarray(q_goal)
        q0 = self.robot.get_dofs_position().cpu().numpy()
        for i in range(1, steps + 1):
            s = i / steps
            s = s * s * (3 - 2 * s)
            q = q0[:7] + (q_goal[:7] - q0[:7]) * s
            self.robot.control_dofs_position(q, self.arm)
            self.step()
        self.hold(15)

    def open_gripper(self, steps=30):
        cur = self.robot.get_dofs_position().cpu().numpy()[7:9]
        for i in range(1, steps + 1):                   # ramp: abrupt opening kicks the cubes
            q = cur + (OPEN - cur) * (i / steps)
            self.robot.control_dofs_position(q, self.fingers)
            self.step()

    def close_gripper(self, steps=40, force=8.0):
        self.robot.control_dofs_force(np.array([-force, -force]), self.fingers)
        self.hold(steps)

    def go_home(self, steps=100):
        q0 = self.robot.get_dofs_position().cpu().numpy()
        for i in range(1, steps + 1):
            s = i / steps
            s = s * s * (3 - 2 * s)
            self.robot.control_dofs_position((q0 + (self.home - q0) * s)[:7], self.arm)
            self.step()

    # ---- primitives -----------------------------------------------------
    def pick(self, name, clear_z=0.30):
        p = self.pos(name)
        self.open_gripper()
        self.move_to((p[0], p[1], clear_z))
        self.move_to((p[0], p[1], p[2] + HAND_OFFSET), steps=80)
        self.close_gripper()
        self.move_to((p[0], p[1], clear_z), steps=100)

    def place_at(self, held, xy, support_top, clear_z=0.30):
        """Lower the held cube so that its bottom touches `support_top`, release, retreat.
        The measured in-hand offset of the cube is compensated (closed-loop)."""
        off = self.pos(held) - self.hand_pos()          # cube centre relative to hand
        hx, hy = xy[0] - off[0], xy[1] - off[1]
        hz = support_top + CUBE / 2 - off[2] + 0.010
        self.move_to((hx, hy, clear_z), steps=100)
        self.move_to((hx, hy, hz), steps=80)
        self.open_gripper(steps=60)
        self.move_to((hx, hy, hz + 0.05), steps=100)   # slow vertical retreat (no tower tipping)
        self.move_to((hx, hy, clear_z), steps=60)
        self.go_home(80)
