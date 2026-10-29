"""Load the FR3 Duo hackathon scene in Isaac Sim 6.2.

Run inside the open Isaac Sim: Window > Script Editor > File > Open this file > Ctrl+Enter.

Why this file and not FR3Duo_withVMkit.usd:
  - FR3Duo_withVMkit.usd is a visual-only robot export (centimetre units, no physics scene,
    no articulation, no rigid bodies, no cameras). It cannot be simulated.
  - The Collected_* folders are incomplete copies (35 referenced assets are missing).
  - Hackathon_FR3Duo_flattened.usd is the complete, self-contained scene: hospital room, box,
    tray, PhysicsScene, both FR3 arm articulations and the head / wrist cameras (Y-up, metres).

The flattened file still points at textures on the original author's machine
(C:/Users/z0045ytr/...). Copies of them are in Collected_Hackathon_FR3Duo_v2/C/Users/z0045ytr,
so they are re-pointed here. All fixes go into the session layer: the .usd file on disk
is never modified, even if you press Save.
"""
import os

from pxr import Sdf, Usd, UsdPhysics

ENV_DIR = (r"C:\Users\User\Desktop\Hack2026\C2_Teach-the-robot_VLA\Environment"
           r"\Environment_Hackathon_2026\Environment_Hackathon_2026")
SCENE = os.path.join(ENV_DIR, "FR3Duo_withVMkit.usd")
AUTHOR_PREFIX = "C:/Users/z0045ytr/"
TEXTURE_ROOT = os.path.join(ENV_DIR, "Collected_Hackathon_FR3Duo_v2", "C", "Users", "z0045ytr")

ROBOT = "/World/FR3Duo_withV_Mkit__1_"
READY_POSE_DEG = [0, -45, 0, -135, 0, 90, 45]   # nominal arm ready pose from the environment doc
GRIPPER_OPEN_DEG = 0.0                          # gripper_ctrl limit is 0 (open) .. 45.8 (closed)
SIM_HZ = 60                                     # the dataset was recorded from a 60 Hz simulation
PLAY = False                                    # True = press Play automatically after loading


def fix_texture_paths(stage):
    fixed, missing = 0, set()
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            if attr.GetTypeName() != Sdf.ValueTypeNames.Asset:
                continue
            value = attr.Get()
            if not value or not value.path.startswith(AUTHOR_PREFIX):
                continue
            local = os.path.join(TEXTURE_ROOT, *value.path[len(AUTHOR_PREFIX):].split("/"))
            if os.path.exists(local):
                attr.Set(Sdf.AssetPath(local.replace("\\", "/")))
                fixed += 1
            else:
                missing.add(value.path)
    return fixed, missing


def set_physics_rate(stage):
    # Plain attribute authoring instead of PhysxSchema.PhysxSceneAPI, so this also works
    # where the PhysX schema plugin is not registered.
    scene = stage.GetPrimAtPath("/World/PhysicsScene")
    scene.AddAppliedSchema("PhysxSceneAPI")
    scene.CreateAttribute("physxScene:timeStepsPerSecond", Sdf.ValueTypeNames.UInt).Set(SIM_HZ)


def set_ready_pose(stage):
    joints = {p.GetName(): p for p in Usd.PrimRange(stage.GetPrimAtPath(ROBOT))
              if p.IsA(UsdPhysics.Joint)}
    targets = {}
    for side in ("left", "right"):
        for i, deg in enumerate(READY_POSE_DEG, start=1):
            targets[f"{side}_fr3v2_joint{i}"] = deg
        targets[f"{side}_gripper_ctrl"] = GRIPPER_OPEN_DEG
    for name, deg in targets.items():
        UsdPhysics.DriveAPI.Apply(joints[name], "angular").CreateTargetPositionAttr().Set(float(deg))
    return len(targets)


def configure(stage):
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        fixed, missing = fix_texture_paths(stage)
        set_physics_rate(stage)
        n_targets = set_ready_pose(stage)
    print(f"[fr3duo] textures re-pointed: {fixed}, still missing: {len(missing)}")
    for path in sorted(missing):
        print("[fr3duo]   missing:", path)
    print(f"[fr3duo] physics rate set to {SIM_HZ} Hz")
    print(f"[fr3duo] drive targets set on {n_targets} joints (ready pose, grippers open)")
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            print("[fr3duo] articulation:", prim.GetPath())
        if prim.GetTypeName() == "Camera" and str(prim.GetPath()).startswith(ROBOT):
            print("[fr3duo] camera:", prim.GetPath())


def main():
    import omni.usd
    import omni.timeline

    ctx = omni.usd.get_context()
    if not ctx.open_stage(SCENE):
        raise RuntimeError(f"could not open {SCENE}")
    stage = ctx.get_stage()
    print("[fr3duo] opened", SCENE,
          "| upAxis:", stage.GetMetadata("upAxis"), "| metersPerUnit:", stage.GetMetadata("metersPerUnit"))
    configure(stage)
    if PLAY:
        omni.timeline.get_timeline_interface().play()
    else:
        print("[fr3duo] press Play: the arms move to the ready pose")


main()
