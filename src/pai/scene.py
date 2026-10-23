"""Genesis tabletop scene + Franka Panda controller (IK + joint-space motion)."""
import numpy as np
import genesis as gs

CUBE = 0.04          # cube edge [m]
PLATE_R, PLATE_H = 0.075, 0.012

# id -> (kind, color rgb, initial xy, description)
OBJECTS = {
    "red_cube":    ("cube",  (0.90, 0.10, 0.10), (0.50, 0.18)),
    "green_cube":  ("cube",  (0.10, 0.75, 0.20), (0.42, -0.05)),
    "blue_cube":   ("cube",  (0.15, 0.30, 0.95), (0.60, 0.00)),
    "yellow_cube": ("cube",  (0.95, 0.85, 0.10), (0.38, 0.25)),
    "blue_plate":  ("plate", (0.20, 0.45, 0.85), (0.50, -0.22)),
}
DESCRIPTIONS = {
    "red_cube": "red cube (赤いキューブ)", "green_cube": "green cube (緑のキューブ)",
    "blue_cube": "blue cube (青いキューブ)", "yellow_cube": "yellow cube (黄色いキューブ)",
    "blue_plate": "blue plate (青い皿), fixed on the table",
}
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
        self.scene.add_entity(gs.morphs.Plane())
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
        self.ent = {}
        for name, (kind, col, (x, y)) in OBJECTS.items():
            surf = gs.surfaces.Default(color=col)
            if kind == "cube":
                self.ent[name] = self.scene.add_entity(
                    gs.morphs.Box(size=(CUBE,) * 3, pos=(x, y, CUBE / 2)),
                    material=gs.materials.Rigid(friction=2.0), surface=surf)
            else:
                self.ent[name] = self.scene.add_entity(
                    gs.morphs.Cylinder(radius=PLATE_R, height=PLATE_H,
                                       pos=(x, y, PLATE_H / 2), fixed=True), surface=surf)
        self.cam = self.scene.add_camera(res=res, pos=(1.45, -0.95, 1.0),
                                         lookat=(0.38, 0.0, 0.15), fov=45, GUI=False)
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
            q = cur + (0.04 - cur) * (i / steps)
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

    def teleport(self, name, xyz):
        """External disturbance (used to demo failure recovery)."""
        self.ent[name].set_pos(np.asarray(xyz, dtype=float))
        self.hold(20)

    def reset(self):
        """Put all objects back to their initial poses and reopen the gripper."""
        for name, (kind, _, (x, y)) in OBJECTS.items():
            if kind == "cube":
                self.ent[name].set_pos(np.array([x, y, CUBE / 2]))
                self.ent[name].set_quat(np.array([1.0, 0, 0, 0]))
        self.open_gripper()
        self.hold(30)
