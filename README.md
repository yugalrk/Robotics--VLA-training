# FR3 Duo handover: training and evaluation scripts

Siemens Healthineers Hackathon 2026, challenge C2. Scripts to train imitation-learning policies (ACT, SmolVLA)
on the FR3 Duo demonstrations, label the data with sub-tasks, and evaluate policies offline and closed-loop in
Isaac Sim.

**Models and data are not in this repository.** Put them next to the scripts (paths below) before running.

## Expected layout
```
Datasets/fr3_duo_handover_84_ep/fr3_duo_handover_v9_night_84/   handover dataset (LeRobot v3, 84 episodes)
Datasets/fr3_duo_rand500/fr3_duo_rand500/                       cube-stacking dataset (490 episodes)
Environment/.../Hackathon_FR3Duo_flattened.usd                  Isaac Sim scene
outputs/<run>/step_XXXXXX/                                      trained checkpoints (written by training)
```

## Scripts

| Stage | Script | What it does |
|---|---|---|
| Data | `fr3_data.py` | Dataset paths, train/val split (`splits.json`: 10 held-out episodes with extreme/random lighting), loader |
| Data | `label_rand500_colors.py` | Labels each cube episode with cube colours + gripper event frames → `rand500_labels.csv` |
| Data | `make_subtasks.py` | Per-frame sub-task labels from robot signals (LeRobot-native `subtask_index`), quality checks, review images |
| Train | `train_curriculum.py` | **Curriculum** training of SmolVLA / pi0 / pi0.5: basic skills → bimanual → full tasks (see [CURRICULUM.md](CURRICULUM.md)) |
| Train | `train_act_mixed.py` | ACT or SmolVLA (`--policy act/smolvla`) on handover and/or cubes (`--data handover/both`), 6 Hz, own norm stats |
| Train | `train_act.py` | Plain `lerobot-train` wrapper for ACT on the handover data (30 fps), with Windows fixes |
| Serve | `serve_policy.py` | Loads any LeRobot checkpoint and answers observations with joint targets over TCP; `--selftest` |
| Eval | `eval_offline.py` | Held-out episodes without a simulator: error per sub-task, horizon, gripper timing, DTW, smoothness, latency, robustness |
| Eval | `export_eval_assets.py` | Writes `eval_assets/handover_reference.json` (start pose, demo for replay, held-out lighting) |
| Eval | `eval_isaac.py` | Runs inside Isaac Sim: replay check of the demo, then closed-loop policy episodes with stage scoring |
| Misc | `main.py` | Quick look at the raw handover data |

Details on the simulator evaluation and the scene's gaps: [EVAL_README.md](EVAL_README.md).

## Setup (training / policy server side)
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```
Isaac Sim's own Python needs nothing extra for `eval_isaac.py` (numpy + Pillow are bundled).

## Typical workflow
```bash
# train (each run ~25 min on an 8 GB laptop GPU)
python train_act_mixed.py --policy act     --data handover --steps 6000 --out outputs/act_handover
python train_act_mixed.py --policy smolvla --data handover --steps 3000 --batch 8 --out outputs/smolvla_handover

# offline evaluation on the held-out episodes -> eval_results/offline/summary.md
python eval_offline.py --models act_handover/step_006000 smolvla_handover/step_003000

# closed loop in Isaac Sim (server and simulator can be on different machines)
python export_eval_assets.py
python serve_policy.py --checkpoint outputs/smolvla_handover/step_003000 --port 5555
<isaac>/python.sh eval_isaac.py --mode replay --usd <scene.usd> --out eval_results/replay
<isaac>/python.sh eval_isaac.py --mode policy --usd <scene.usd> --server <server-ip>:5555 --episodes 10 --out eval_results/smolvla
```

## Notes
- Policies run at 6 Hz with 20-action chunks (3.3 s); the 30 fps handover data is sampled every 5th frame.
- SmolVLA needs the exact training instruction at inference (`eval_isaac.py` sends it).
- `eval_isaac.py` has not been run on a real Isaac Sim install yet; expect small API fixes on first run.

## Post-training pipeline, UI, serving and Isaac Sim tools

| Area | Files |
|---|---|
| Training additions | `train_curriculum.py`: LoRA (`--lora_rank`), gradient accumulation (`--grad_accum`), crash-safe `--save_every` / `--resume`; `run_smolvla_queue.ps1` |
| Training curves | `plot_training.py` + `plot_training_template.html` -> `outputs/<run>/training_curves.html` |
| Post-training eval | `post_train_eval.py`: offline eval + headless Isaac Sim eval (with a demo-replay sanity check) + gates (`deploy_gates.json`) -> DEPLOY / MORE TRAINING / HOLD |
| Pipeline UI | `pipeline_ui.py` + `pipeline_ui.html` (http://127.0.0.1:8765): runs, evaluate / auto-evaluate, verdicts, deploy with staging + health check, stop / start / rollback serving on port 8090, re-plan and grasp-extension settings |
| Serving for Isaac Sim | `Environment/fr3duo_policy_server.py` (HTTP :8090; `--n_action_steps`, `--grasp_extend`), `Environment/fr3duo_pi05_server.py` (openpi pi0.5, WSL) |
| Isaac Sim (Script Editor) | `Environment/fr3duo_isaac_client.py` (manual runs), `Environment/fr3duo_isaac_eval.py` (scored runs: task physics, held-out lighting, stage score) |
| Diagnostics | `tools/` (grasp timing vs demo, sim-vs-training camera gap, pi0.5 input probes, openpi / gh install in WSL) |

Known sim gap: the hand-out scene lacks the recorder's override layers (`cameras_v9_overrides.usda`,
`rand_v7_boxpose_overrides.usda`, `handover_root_v9_light_boxpose.usda`; listed in the dataset's
`fr3_episodes_extra.jsonl` stack_lock), so even the recorded demo does not grasp the box in the sim.

## Robot model, retargeting, box-grid LUT and demo UIs

| Area | Files |
|---|---|
| Robot model (FK) | `fk_fr3duo.py` + `geometry/fr3duo_scene_chain.json` (extracted from the scene USD by `tools/extract_scene_kinematics.py`; matches Franka's official FR3v2 / FR3 Duo description to 0.000 mm) |
| Head camera model | `tools/head_camera_model.py` -> `geometry/head_camera.json` (scene pose + recorder intrinsics) |
| Retargeting (IK) | `tools/retarget_trajectory.py`: damped least squares, 6-D pose, null space pulled to the recorded joints, joint limits |
| Box-grid dataset + LUT | `tools/augment_box_grid.py` (5 cm grid over the cart top -> ok / out_of_reach / off_cart, writes `Datasets/.../Augmented`), `tools/camera_grid.py` (camera -> cart plane -> grid cell) |
| Trajectory policy (no model) | `Environment/fr3duo_policy_server.py --trajectory`: recorded handover as a closed-loop lookup; `tools/test_trajectory_policy.py` |
| Model probes | `tools_show_io.py` (SmolVLA inputs/outputs), `tools_ablation.py` (camera / state ablations) |
| Demo UIs | `demo_ui/` (handover explorer site, sub-task splitting page, box grid player, XYZ views); see `demo_ui/README.md` |

This highlighted our main challenge: limited diversity in the training data, which motivated us to explore ways to generate more varied demonstrations.

Our Approach: Kinematics-Based Data Augmentation

Our solution reuses recorded robot movements and combines them with the robot's kinematic model to calculate new trajectories for different box positions, without training an additional AI model.

1. Data Exploration

We analyze the existing demonstrations, including camera recordings, joint states, and actions. Each robot state contains 16 values: seven joint angles and one gripper state per arm.

These recordings provide the original movement patterns used as references for trajectory generation.

2. Robot Kinematics and Trajectory Reconstruction

We combine the recorded joint angles with the FR3 Duo's kinematic model, which describes its link geometry, joint axes, and movement limits.

Two main algorithms are used:

Forward Kinematics (FK): Reconstructs the robot's original gripper positions and trajectories in 3D space from recorded joint angles.

Inverse Kinematics (IK): Calculates joint configurations needed to reach new target positions.

Our IK implementation uses Damped Least Squares (DLS) with joint-limit constraints and null-space optimization to keep calculated joint configurations close to the original recorded movements.

This allows us to estimate the robot's reachable workspace and adapt movements to new box locations while preserving the reference trajectory as much as possible.

3. Trajectory Retargeting and Lookup Table

We divide the tabletop into a 5 cm grid and calculate candidate left-arm trajectories for different box positions.

The process involves:

Mapping the selected box position from camera coordinates to the tabletop grid.

Adapting the original gripper trajectory to the new target.

Using inverse kinematics to calculate the corresponding joint movements.

Precomputing and storing the resulting trajectories in a Lookup Table (LUT) for fast retrieval.

Current results:

136 feasible grid positions with candidate left-arm trajectories.

26 grid positions marked as outside the reachable workspace under the implemented constraints.

Interactive visualization of the calculated trajectories for different box positions.

4. Toward Automated Training Data Generation

The next step is to execute and validate the adapted trajectories in Isaac Sim, including grasp success, collision checks, and the complete dual-arm handover.

Successful executions could then be recorded with synchronized camera observations, robot states, and actions to create new training demonstrations covering more diverse box positions.

These augmented demonstrations could be used to fine-tune imitation-learning models and evaluate whether increased data diversity improves generalization.
