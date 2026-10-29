# Evaluating a policy in Isaac Sim (team workspace)

The policy and the simulator run as two separate processes:

```
serving/serve_policy.py  (LeRobot + torch, GPU)  <── TCP :5555 ──>  isaac/eval_isaac.py  (Isaac Sim Python, numpy only)
```

## Files to copy to the workspace
- `serving/serve_policy.py`, `fr3duo/data.py`, `configs/splits.json`
- `isaac/eval_isaac.py`, `artifacts/eval_assets/handover_reference.json`
- the checkpoint folder, e.g. `outputs/act_mixed/step_006000/`
- the scene: `Environment/.../Hackathon_FR3Duo_flattened.usd` (self-contained, 57 MB)

## 1. Start the policy server (any Python with `lerobot==0.4.4` and CUDA torch)
```
python serving/serve_policy.py --checkpoint outputs/act_mixed/step_006000 --port 5555
```
Options: `--n_action_steps 5` re-plans more often (default executes the whole 20-step chunk, 3.3 s);
`--temporal_ensemble 0.01` re-plans every step and blends overlapping chunks (smoother, slower).
Check the checkpoint without a simulator: `--selftest`.

## 2. Check the sim setup first: replay the demo
```
<isaac-sim>/python.sh isaac/eval_isaac.py --mode replay --usd <path>/Hackathon_FR3Duo_flattened.usd --out eval_results/replay
```
Plays the recorded actions open-loop. Look at `eval_results/replay/replay.json`:
- `success: true` and low `joint_tracking_rad` → the sim matches the recording; policy results are meaningful.
- box not on tray → the scene differs from the recording setup (the organisers used extra override layers
  that are not in the hand-out: box pose, contact sensing, gripper drive). Try `--gripper_drive usd`, and ask
  the mentors for `handover_root_v9_light_boxpose.usda`.
- `gripper_fit_state_eq` shows how the sim's gripper joint maps to the dataset's gripper state. If it is far
  from the default mapping in `gripper_joint_to_state`, update that function.

## 3. Evaluate the policy
```
<isaac-sim>/python.sh isaac/eval_isaac.py --mode policy --usd <...> --server 127.0.0.1:5555 --episodes 10 --out eval_results/act_mixed_6k
```
Each episode uses the lighting of one held-out episode (never seen in training). Outputs:
- `results.json`: per episode `success`, `stage` (0 not lifted · 1 lifted · 2 carried halfway · 3 over tray ·
  4 resting on tray), final box position, distance to the demo's final box pose, tilt, latency (median/p95)
- `policy_epXX.gif`: head | left wrist | right wrist camera at 6 Hz

## Which scene file
Use `Hackathon_FR3Duo_flattened.usd`. Checked against the other files in the hand-out:
- `FR3Duo_withVMkit.usd` is the robot's 3D shape only (no joints, cameras or physics; needs a missing `props/` folder).
- `Collected_Hackathon_FR3Duo_v2/Hackathon_FR3Duo.usd` has the rigged robot, but the box has no geometry (its source file is missing).
- The flattened file has the rigged robot, the 3 cameras and the box with collision, all in one file.

Gaps that `isaac/eval_isaac.py` fills in, because the organisers' task settings aren't in the hand-out:
- box mass 0.10 kg and friction 0.9/0.8 (the file has mass 0)
- the soft gripper drive from the recording metadata (`--gripper_drive usd` keeps the file's stiff drive)
- a collider for the serving tray: invisible floor + 4 walls (the tray mesh has none). Turn it off with `--no_tray_collider`.

Not fixable here: 136 textures are missing. Floor, walls, doors and the bed used files on the organiser's PC.
Room props come from NVIDIA's online library and load only if the workspace has internet.
The cameras may therefore see a different-looking room than in training. Ask the mentors for the textures.

## Notes
- The policy runs at 6 Hz; joint targets are linearly interpolated over the 10 physics ticks in between.
- Only the handover task can be evaluated here; the cube-stacking scene is not part of the hand-out.
- Add `--headless false` to watch the robot live.
