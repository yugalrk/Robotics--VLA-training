"""Forward kinematics of the FR3 Duo: joint angles (dataset observation.state) -> gripper-centre XYZ in the room.

    from fr3duo.kinematics import FK
    fk = FK()                      # chain from geometry/fr3duo_scene_chain.json (augmentation/extract_scene_kinematics.py)
    xyz = fk.tcp("left", q7)       # q7 = the arm's 7 joint angles in rad -> (3,) world position, Y-up, metres

Each joint: T_child = T_parent * F(pos0, rot0) * Rot_axis(q) * F(pos1, rot1)^-1, the USD joint definition; the arm
base (link0) is fixed on the mount. `python -m fr3duo.kinematics` runs the self-checks.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AX = {"X": 0, "Y": 1, "Z": 2}


def quat_to_R(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def frame(pos, rot):
    m = np.eye(4)
    m[:3, :3] = quat_to_R(*rot)
    m[:3, 3] = pos
    return m


def rot(axis, q):
    c, s, m = np.cos(q), np.sin(q), np.eye(4)
    i, j = [(1, 2), (2, 0), (0, 1)][AX[axis]]
    m[i, i], m[i, j], m[j, i], m[j, j] = c, -s, s, c
    return m


class FK:
    def __init__(self, path=ROOT / "geometry" / "fr3duo_scene_chain.json"):
        self.chain = json.loads(Path(path).read_text())["arms"]
        for arm in self.chain.values():
            arm["_T0"] = np.array(arm["link0_world"])
            arm["_J"] = [(frame(j["pos0"], j["rot0"]), j["axis"], np.linalg.inv(frame(j["pos1"], j["rot1"]))) for j in arm["joints"]]

    def link7(self, side, q):
        arm = self.chain[side]
        T = arm["_T0"].copy()
        for (F0, axis, F1inv), qi in zip(arm["_J"], q):
            T = T @ F0 @ rot(axis, qi) @ F1inv
        return T

    def tcp(self, side, q):
        return (self.link7(side, q) @ np.r_[self.chain[side]["tcp_in_link7"], 1])[:3]

    def tcp_path(self, side, Q):
        return np.array([self.tcp(side, q) for q in Q])


def franka_dh_flange(q, kin_yaml=ROOT / "geometry/franka_description/robots__fr3v2__kinematics.yaml"):
    """Official Franka FR3v2 chain (URDF style: origin xyz/rpy, then rotate about local z): link0 -> flange."""
    import yaml

    k = yaml.safe_load(Path(kin_yaml).read_text())

    def rpy(r, p, y):
        Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
        Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
        Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
        return Rz @ Ry @ Rx
    T = np.eye(4)
    for i in range(1, 9):
        e = k[f"joint{i}"]["kinematic"]
        o = np.eye(4)
        o[:3, :3], o[:3, 3] = rpy(e["roll"], e["pitch"], e["yaw"]), [e["x"], e["y"], e["z"]]
        T = T @ o @ (rot("Z", q[i - 1]) if i <= 7 else np.eye(4))
    return T


if __name__ == "__main__":
    fk = FK()
    for side in ("left", "right"):
        a = fk.chain[side]
        err = np.abs(fk.link7(side, a["q_authored"]) - np.array(a["link7_world_authored"])).max()
        print(f"[1] {side}: FK(authored angles) vs the scene's own link7 pose: max diff {err:.2e}")
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        q = rng.uniform(-1.5, 1.5, 7)
        T0 = fk.chain["left"]["_T0"]
        scene = (np.linalg.inv(T0) @ fk.link7("left", q) @ np.r_[0, 0, 0.107, 1])[:3]  # flange, in the arm-base frame
        worst = max(worst, np.abs(scene - franka_dh_flange(q)[:3, 3]).max())
    print(f"[2] scene chain vs official Franka FR3v2 kinematics, 200 random poses: max flange position diff {worst * 1000:.3f} mm")
