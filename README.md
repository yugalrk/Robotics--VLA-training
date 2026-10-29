# FR3 Duo handover: teach the robot

Siemens Healthineers Hackathon 2026, challenge C2. The whole learning loop for the FR3 Duo two-arm robot:
ingest the demonstrations, label and augment them, fine-tune imitation-learning policies (ACT, SmolVLA, pi0.5),
evaluate them offline / in MuJoCo / in Isaac Sim, and version + serve them behind deploy gates, with local web
UIs for every step.

**Models and data are not in this repository** (restricted hand-out material). Put them in the folders listed
under [Data and generated folders](#data-and-generated-folders) before running.

## Quick start

```powershell
python -m venv .venv; .venv\Scripts\activate
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .                                   # optional: makes `import fr3duo` work from anywhere

python run_local_demo.py                           # build missing pages + start every local UI
python run_local_demo.py --serve trajectory        # + recorded handover as a lookup policy on :8090
python run_local_demo.py --serve outputs\smolvla_handover\step_003000   # + a trained policy on :8090
```

| Port | Page |
|---|---|
| http://127.0.0.1:8770/ | **Presentation**: the whole project as slides, every demo below embedded live (opens first) |
| http://127.0.0.1:8765/ | Pipeline UI: runs, evaluate, verdicts, deploy / stop / rollback the policy server |
| http://127.0.0.1:8779/ | Handover explorer: `/1-data/`, `/2-model/` (robot model, IK), `/3-lut/` (camera -> grid -> trajectory) |
| http://127.0.0.1:8780/ | Sub-task splitting page (how `data_prep/make_subtasks.py` cuts an episode) |
| http://127.0.0.1:8778/ui/ | Box-grid player: drag the box, see the retargeted trajectory (136 reachable cells) |
| http://127.0.0.1:8777/ | Results browser (`eval_results/`): `/xyz_verify/`, `/mujoco/...` videos + traces, offline summaries |
| http://127.0.0.1:8090/info | Policy server (only with `--serve`), the endpoint the Isaac Sim scripts call |

The first run builds the generated pages (~7 min, mostly the box grid); later runs start in seconds. `--rebuild`
forces a rebuild. Logs: `logs/local_demo_<port>.log`. Ctrl+C stops everything.

All commands run from the repo root. Every script finds the repo root from its own location, so no install step
is needed beyond `requirements.txt`.

## Repository layout

```
fr3duo/            shared code (importable package)
  data.py            dataset paths, train/val split, LeRobot loader
  kinematics.py      forward kinematics of both arms (= official Franka FR3v2 geometry, 0.000 mm)
  retarget.py        IK retargeting of the recorded left-arm motion to a moved box
  head_camera.py     head-camera model: world <-> pixel, pixel -> table plane
  policy_runner.py   load and run any LeRobot checkpoint (+ TCP server)
  subtasks.py        per-frame sub-task segmentation + quality checks

data_prep/         1. ingest + dataset intelligence
augmentation/      2. FK/IK box-position grid (new trajectories without a model)
training/          3. fine-tuning (ACT, SmolVLA, curriculum, LoRA) + training curves
evaluation/        4. offline eval, post-training eval + deploy gates
sim_mujoco/        4. closed-loop eval in a MuJoCo replica of the scene (runs on a laptop)
isaac/             4. closed-loop eval inside Isaac Sim (Isaac's own Python, no project imports)
serving/           5. policy servers (HTTP :8090 for Isaac Sim, TCP :5555, openpi pi0.5 in WSL)
ui/pipeline/       5. pipeline UI: evaluate, verdicts, deploy / rollback
ui/demo/           explainer pages + static file server
diagnostics/       one-off probes (grasp timing, camera domain gap, pi0.5 inputs, ablations)

configs/           splits.json (held-out episodes), deploy_gates.json
geometry/          robot chain + head camera extracted from the scene USD
docs/              CURRICULUM.md, EVALUATION.md, APPROACH.md (kinematics augmentation)
notebooks/         exploration notebook
run_local_demo.py  one command for the whole local demo
```

## Data and generated folders

Not in git. Inputs (copy these in):

```
Datasets/fr3_duo_handover_84_ep/fr3_duo_handover_v9_night_84/   handover dataset (LeRobot v3, 84 episodes)
Datasets/fr3_duo_rand500/fr3_duo_rand500/                       cube-stacking dataset (490 episodes)
Environment/Environment_Hackathon_2026/Environment_Hackathon_2026/Hackathon_FR3Duo_flattened.usd   Isaac Sim scene
```

Written by the scripts:

| Folder | Contents |
|---|---|
| `Datasets/fr3_handover84_subtasks/`, `Datasets/fr3_rand500_subtasks/` | datasets with sub-task labels (`data_prep/make_subtasks.py`) |
| `Datasets/fr3_duo_handover_84_ep/Augmented/` | box-grid trajectories + player (`augmentation/augment_box_grid.py`) |
| `outputs/<run>/` | checkpoints, training logs, curves |
| `eval_results/` | offline / MuJoCo / Isaac results, pipeline verdicts |
| `artifacts/` | small derived data: `eval_assets/handover_reference.json`, `subtask_qa/`, `rand500_labels.csv` |
| `build/` | generated web pages (`site/`, `subtask_demo/`) |
| `deploy/` | which checkpoint is in production (pipeline UI) |
| `logs/` | server and run logs |
| `third_party/` | `mujoco_menagerie` (MuJoCo robot models), `openpi` (pi0.5) |

## Scripts by stage

| Stage | Script | What it does |
|---|---|---|
| Data | `data_prep/quick_look.py` | Quick look at the raw handover data |
| Data | `data_prep/make_splits.py` | `configs/splits.json`: 10 held-out episodes with extreme/random lighting |
| Data | `data_prep/label_rand500_colors.py` | Cube colours + gripper event frames per cube episode -> `artifacts/rand500_labels.csv` |
| Data | `data_prep/make_subtasks.py` | Per-frame sub-task labels from robot signals (LeRobot-native `subtask_index`), quality checks, review images |
| Augment | `augmentation/extract_scene_kinematics.py` | Robot chain from the scene USD -> `geometry/fr3duo_scene_chain.json` |
| Augment | `augmentation/head_camera_model.py` | Head camera pose + intrinsics -> `geometry/head_camera.json` |
| Augment | `augmentation/retarget_trajectory.py` | One box shift: Cartesian shift + damped-least-squares IK, viewer page |
| Augment | `augmentation/augment_box_grid.py` | 5 cm grid over the cart: ok / out_of_reach / off_cart, writes `Augmented/` |
| Augment | `augmentation/camera_grid.py` | Camera pixel -> cart plane -> grid cell (LUT lookup from the head camera) |
| Train | `training/train_curriculum.py` | **Curriculum** training of SmolVLA / pi0 / pi0.5 (skills -> bimanual -> full task), LoRA, grad accumulation, `--resume`; see [docs/CURRICULUM.md](docs/CURRICULUM.md) |
| Train | `training/train_act_mixed.py` | ACT or SmolVLA on handover and/or cubes, 6 Hz, own norm stats |
| Train | `training/train_act.py` | Plain `lerobot-train` wrapper for ACT (30 fps), with Windows fixes |
| Train | `training/plot_training.py` | `outputs/<run>/training_curves.html` |
| Train | `training/run_smolvla_queue.ps1` | Crash-safe low-load SmolVLA curriculum run |
| Eval | `evaluation/eval_offline.py` | Held-out episodes: error per sub-task, horizon, gripper timing, DTW, smoothness, latency, robustness |
| Eval | `evaluation/export_eval_assets.py` | `artifacts/eval_assets/handover_reference.json` (start pose, demo, held-out lighting) |
| Eval | `evaluation/post_train_eval.py` | Offline + headless Isaac eval + gates (`configs/deploy_gates.json`) -> DEPLOY / MORE TRAINING / HOLD |
| Eval | `evaluation/eval_smolvla_loop.py` | SmolVLA closed on its own predictions over recorded frames |
| Eval | `sim_mujoco/eval_mujoco.py` | MuJoCo replica: `--mode replay|policy`, videos with status overlay + traces |
| Eval | `isaac/eval_isaac.py` | Inside Isaac Sim: demo replay check, then closed-loop episodes with stage scoring |
| Eval | `isaac/fr3duo_isaac_client.py`, `isaac/fr3duo_isaac_eval.py` | Isaac Script Editor: manual runs / scored runs against the :8090 server |
| Serve | `serving/fr3duo_policy_server.py` | HTTP :8090 for Isaac Sim (`--checkpoint`, `--trajectory`, `--dummy`, `--n_action_steps`, `--grasp_extend`) |
| Serve | `serving/serve_policy.py` | TCP :5555 for `isaac/eval_isaac.py`; `--selftest` |
| Serve | `serving/fr3duo_pi05_server.py` + `serving/wsl_*.sh` | openpi pi0.5 in WSL, same HTTP protocol |
| UI | `ui/pipeline/pipeline_ui.py` | Pipeline UI on :8765 |
| UI | `ui/demo/` | Explainer page builders + `serve_static.py`; see [ui/demo/README.md](ui/demo/README.md) |

## Typical workflow

```bash
# data
python data_prep/make_splits.py
python data_prep/label_rand500_colors.py
python data_prep/make_subtasks.py

# train (each run ~25 min on an 8 GB laptop GPU)
python training/train_act_mixed.py --policy act     --data handover --steps 6000 --out outputs/act_handover
python training/train_act_mixed.py --policy smolvla --data handover --steps 3000 --batch 8 --out outputs/smolvla_handover
python training/train_curriculum.py --policy smolvla --steps 20000 --batch 32 --out outputs/curriculum_smolvla

# evaluate
python evaluation/eval_offline.py --models act_handover/step_006000 smolvla_handover/step_003000
python sim_mujoco/eval_mujoco.py --mode policy --checkpoint outputs/smolvla_handover/step_003000
python evaluation/post_train_eval.py --checkpoint outputs/<run>/final          # offline + Isaac + gates

# closed loop in Isaac Sim (server and simulator can be on different machines)
python evaluation/export_eval_assets.py
python serving/serve_policy.py --checkpoint outputs/smolvla_handover/step_003000 --port 5555
<isaac>/python.sh isaac/eval_isaac.py --mode replay --usd <scene.usd> --out eval_results/replay
<isaac>/python.sh isaac/eval_isaac.py --mode policy --usd <scene.usd> --server <server-ip>:5555 --episodes 10 --out eval_results/smolvla
```

Details on the simulator evaluation and the scene's gaps: [docs/EVALUATION.md](docs/EVALUATION.md).

## MuJoCo replica

`sim_mujoco/` rebuilds the FR3 Duo scene in MuJoCo from the Isaac USD (Menagerie FR3 + Robotiq 2F-85, link poses
match the USD to 0.00 mm; joint-1 sag and the recorded box pose reconstructed from the demo), so the recorded demo
replays to success. One-time setup:

```bash
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie third_party/mujoco_menagerie
```

## Notes

- Policies run at 6 Hz with 20-action chunks (3.3 s); the 30 fps handover data is sampled every 5th frame.
- SmolVLA needs the exact training instruction at inference (the eval and server scripts send it).
- Known sim gap: the hand-out scene lacks the recorder's override layers (`cameras_v9_overrides.usda`,
  `rand_v7_boxpose_overrides.usda`, `handover_root_v9_light_boxpose.usda`; listed in the dataset's
  `fr3_episodes_extra.jsonl` stack_lock), so even the recorded demo does not grasp the box in Isaac Sim.
- Why we built the box-grid augmentation, and what it gives: [docs/APPROACH.md](docs/APPROACH.md).
