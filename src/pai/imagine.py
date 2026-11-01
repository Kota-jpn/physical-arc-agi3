"""Imagination: parallel copies of the arm in which a cube's mass is a hypothesis.

The robot cannot see weight. When it lifts a cube, its joint torques change, but how much depends on the arm
pose and the grasp. So instead of a hand-made formula, the same pick is replayed in K simulated worlds that differ
only in the cube mass (one Genesis scene, K parallel environments), starting from the real arm state. The world
whose torques match the real reading best gives the felt weight. The true mass is never read: only the torques.
"""
import numpy as np
import genesis as gs

from .grid import CUBE
from .scene import GRIP, HAND_OFFSET, OPEN, QUAT_DOWN, HIDDEN, RIGID

MASSES = (0.04, 0.1, 0.2, 0.3, 0.5, 0.8)      # mass hypotheses [kg], one world each
SPACING = 0.9                                  # distance between the worlds [m]


class Imagination:
    def __init__(self, sim, res=(1440, 540)):
        k = len(MASSES)
        self.scene = gs.Scene(
            show_viewer=False,
            sim_options=gs.options.SimOptions(dt=0.01, substeps=2),
            vis_options=gs.options.VisOptions(show_world_frame=False),
            rigid_options=gs.options.RigidOptions(batch_links_info=True, **RIGID))
        self.scene.add_entity(gs.morphs.Plane(), surface=gs.surfaces.Default(color=(0.08, 0.08, 0.12)))
        self.robot = self.scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
        self.cube = self.scene.add_entity(gs.morphs.Box(size=(CUBE,) * 3, pos=(0.5, 0, CUBE / 2)),
                                          material=gs.materials.Rigid(friction=2.0),
                                          surface=gs.surfaces.Default(color=(1.0, 0.86, 0.0)))
        self.pedestal = self.scene.add_entity(gs.morphs.Box(size=(CUBE, CUBE, 0.2), pos=(0.5, 0, HIDDEN - 0.1), fixed=True),
                                              surface=gs.surfaces.Default(color=(0.3, 0.3, 0.3)))
        self.cam = self.scene.add_camera(res=res, pos=(0.25, -2.4, 0.9), lookat=(0.25, 0.0, 0.2), fov=40, GUI=False)
        self.scene.build(n_envs=k, env_spacing=(SPACING, SPACING), n_envs_per_row=1)
        self.cube.set_mass(np.array(MASSES))
        self.k, self.hand = k, self.robot.get_link("hand")
        self.arm, self.fingers = np.arange(7), np.array([7, 8])
        self.robot.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
        self.robot.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
        self.robot.set_dofs_force_range(np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
                                        np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]))
        off = np.asarray(self.scene.envs_offset)
        self.offsets = off - off[0]                         # where each copy is drawn, relative to world 0 (for labels)
        cx = float(self.offsets[:, 0].mean()) + 0.15
        self.cam_pose = ((cx, -3.1, 1.0), (cx, 0.0, 0.25), 40)
        self.cam.set_pose(pos=self.cam_pose[0], lookat=self.cam_pose[1])
        self.frames, self.record = [], False

    def _tile(self, v):
        return np.tile(np.asarray(v, dtype=float), (self.k, 1))

    def _step(self, n=1):
        for _ in range(n):
            self.scene.step()
            self.n += 1
            if self.record and self.n % 8 == 0:
                self.frames.append(self.cam.render()[0])

    # the same motions as Sim.pick, for all worlds at once
    def _move_to(self, xyz, steps=120):
        q_goal = self.robot.inverse_kinematics(link=self.hand, pos=self._tile(xyz), quat=self._tile(QUAT_DOWN))
        q_goal = q_goal.cpu().numpy()
        q0 = self.robot.get_dofs_position().cpu().numpy()
        for i in range(1, steps + 1):
            s = i / steps
            s = s * s * (3 - 2 * s)
            self.robot.control_dofs_position(q0[:, :7] + (q_goal[:, :7] - q0[:, :7]) * s, self.arm)
            self._step()
        self._step(15)

    def weigh(self, q0, cube_pos, cube_quat, real_tau, clear_z=0.30, record=False):
        """Replay a pick from arm state `q0` of a cube at `cube_pos` / `cube_quat` in every world; compare the
        joint torques with `real_tau`. Returns (estimated mass [kg], index of the closest world, torques of all worlds)."""
        self.record, self.frames, self.n = record, [], 0
        self.robot.set_qpos(self._tile(q0))
        self.robot.control_dofs_position(self._tile(q0[:7]), self.arm)
        self.robot.control_dofs_position(self._tile(q0[7:9]), self.fingers)
        p = np.asarray(cube_pos, dtype=float)
        top = p[2] - CUBE / 2                                 # a stacked cube rests on a pedestal of that height
        self.pedestal.set_pos(self._tile((p[0], p[1], top - 0.1 if top > 0.005 else HIDDEN - 0.1)))
        self.cube.set_pos(self._tile(p))
        self.cube.set_quat(self._tile(cube_quat))
        self.cube.set_dofs_velocity(np.zeros((self.k, 6)))
        cur = q0[7:9]
        for i in range(1, 31):                                # open_gripper
            self.robot.control_dofs_position(self._tile(cur + (OPEN - cur) * (i / 30)), self.fingers)
            self._step()
        self._move_to((p[0], p[1], clear_z))
        self._move_to((p[0], p[1], p[2] + HAND_OFFSET), steps=80)
        self.robot.control_dofs_force(self._tile((-GRIP, -GRIP)), self.fingers)
        self._step(40)
        self._move_to((p[0], p[1], clear_z), steps=100)
        taus = []
        for i in range(25):                                   # held still; average the last 10 steps as the real arm does
            self._step()
            if i >= 15:
                taus.append(self.robot.get_dofs_control_force().cpu().numpy()[:, :7])
        taus = np.mean(taus, axis=0)
        self.record = False
        return (*estimate(taus, np.asarray(real_tau)), taus)


LOADED = [1, 3, 5]      # pitch joints: carry the weight of the load; the others mostly see grasp details


def estimate(taus, real):
    """Mass whose torques lie closest to `real`, interpolating along the curve of the worlds."""
    taus, real = taus[:, LOADED], real[LOADED]
    best = (np.inf, MASSES[0])
    for i in range(len(MASSES) - 1):
        a, b = taus[i], taus[i + 1]
        t = float(np.clip(np.dot(real - a, b - a) / max(np.dot(b - a, b - a), 1e-12), 0, 1))
        d = float(np.linalg.norm(real - (a + t * (b - a))))
        if d < best[0]:
            best = (d, MASSES[i] + t * (MASSES[i + 1] - MASSES[i]))
    mass = best[1]
    return mass, int(np.argmin([abs(m - mass) for m in MASSES]))
