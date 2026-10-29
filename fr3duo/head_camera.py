"""Head-camera model for the recorded handover video: pose from the scene USD, intrinsics from the recorder metadata.

    (USD libs on PYTHONPATH) .venv\\Scripts\\python.exe augmentation\\head_camera_model.py   -> geometry/head_camera.json

Recorder (meta/fr3_conversion.json): head_cam focal 2.4 mm, apertures 4.8 x 4.8 mm, 224 x 224 px
  -> fx = fy = 2.4 / 4.8 * 224 = 112 px, principal point at the image centre.
USD camera convention: looks along its local -Z, local +Y is up. world_to_pixel(): u right, v down.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAM = "/World/FR3Duo_withV_Mkit__1_/mount_link/cover_link/head_link/zed_mini_camera_link/zed_mini_left_camera_frame/head_cam"


def load(path=ROOT / "geometry/head_camera.json"):
    return json.loads(Path(path).read_text())


def world_to_pixel(cam, P):
    """P: (N, 3) world points -> (N, 2) pixels (u, v) and depth along the view direction."""
    T = np.linalg.inv(np.array(cam["world_from_camera"]))
    Pc = (T @ np.c_[P, np.ones(len(P))].T).T[:, :3]
    z = -Pc[:, 2]
    u = cam["cx"] + cam["fx"] * Pc[:, 0] / z
    v = cam["cy"] - cam["fy"] * Pc[:, 1] / z
    return np.c_[u, v], z


def pixel_to_plane(cam, uv, height):
    """Back-project pixels onto the horizontal plane y = height (world). Returns (N, 3) world points."""
    W = np.array(cam["world_from_camera"])
    uv = np.atleast_2d(uv)
    d_cam = np.c_[(uv[:, 0] - cam["cx"]) / cam["fx"], -(uv[:, 1] - cam["cy"]) / cam["fy"], -np.ones(len(uv))]
    d = (W[:3, :3] @ d_cam.T).T
    o = W[:3, 3]
    s = (height - o[1]) / d[:, 1]
    return o + s[:, None] * d


def main():
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(ROOT / "Environment/Environment_Hackathon_2026/Environment_Hackathon_2026/Hackathon_FR3Duo_flattened.usd"))
    prim = stage.GetPrimAtPath(CAM)
    M = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(prim)).T  # column-vector convention
    cam = {"prim": CAM, "world_from_camera": M.tolist(), "fx": 112.0, "fy": 112.0, "cx": 112.0, "cy": 112.0,
           "width": 224, "height": 224, "source": "pose: scene USD (authored); intrinsics: recorder metadata (focal 2.4 mm, aperture 4.8 mm)"}
    out = ROOT / "geometry/head_camera.json"
    out.write_text(json.dumps(cam, indent=1))
    print("camera at", np.round(M[:3, 3], 4).tolist(), "looking along", np.round(-M[:3, 2], 3).tolist(), "->", out)


if __name__ == "__main__":
    main()
