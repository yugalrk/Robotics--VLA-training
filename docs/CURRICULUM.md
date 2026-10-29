# Curriculum training of a VLA on FR3 Duo sub-tasks

Teach basic skills first, then harder ones, then the complete long tasks, on the two datasets in hand:
the clinical **handover** (84 episodes) and the **cube stacking** (490 episodes).

## Pipeline (run in this order on the training machine)

```bash
# 0. data in place (not in the repo):
#    Datasets/fr3_duo_handover_84_ep/fr3_duo_handover_v9_night_84/
#    Datasets/fr3_duo_rand500/fr3_duo_rand500/

# 1. cube colours + gripper events per cube episode            -> artifacts/rand500_labels.csv            (~3 min)
python data_prep/label_rand500_colors.py

# 2. sub-task labels + quality checks + review images            -> Datasets/fr3_handover84_subtasks/
#    (videos are hard-linked, nothing re-encoded)                   Datasets/fr3_rand500_subtasks/
python data_prep/make_subtasks.py                                           #  artifacts/subtask_qa/*  (~5 min)
#    check artifacts/subtask_qa/*_qa.json (flagged episodes) and *_boundaries_*.png before training

# 3. curriculum training
python training/train_curriculum.py --policy smolvla --steps 20000 --batch 32 --out outputs/curriculum_smolvla
# baseline for comparison: same data and steps, all skills mixed from the start
python training/train_curriculum.py --policy smolvla --steps 20000 --batch 32 --no_curriculum --out outputs/mixed_smolvla

# 4. evaluate (each stage's checkpoint works with serving/serve_policy.py / evaluation/eval_offline.py / isaac/eval_isaac.py)
python evaluation/eval_offline.py --models curriculum_smolvla/final mixed_smolvla/final
```

## Sub-tasks and skills

Handover sub-tasks (default, `data_prep/make_subtasks.py --handover_subtasks 3`):

| # | Sub-task | Frames @30 fps | Skill |
|---|---|---|---|
| 1 | Pick up the box with the left gripper. | 0 – 338 (11.3 s) | `reach_grasp` |
| 2 | Hand the box over to the right gripper. | 338 – 1482 (38.1 s) | `bimanual` |
| 3 | Place the box on the serving tray. | 1482 – 2247 (25.5 s) | `transport_place` |

`--handover_subtasks 5` splits step 2 into *Bring the box to the handover position* · *Grasp the box with the
right gripper* · *Release the box from the left gripper* (finer control, same boundaries otherwise).
Note: "pick up" ends when the grasp is complete; the lift starts the hand-over step, because the demo lifts and
carries in one continuous motion with no pause in between.

| Skill (difficulty) | Handover | Cubes |
|---|---|---|
| 1 `reach_grasp`: one arm picks an object | Pick up the box with the left gripper | Pick up the {A} cube with the left gripper · Pick up the {C} cube with the right gripper |
| 2 `transport_place`: one arm carries / puts it | Place the box on the serving tray | Put the {A} cube on the {B} cube · Stack the {C} cube on top of the {A} cube |
| 3 `bimanual`: both arms coordinate | Hand the box over to the right gripper | — |
| 4 `full_task`: whole episode, complete instruction | Hand the box from the left gripper to the right gripper and place it on the serving tray | Stack the {A} cube on the {B} cube, then put the {C} cube on top |

Skills are assigned from the sub-task text (`SKILL_RULES` in `training/train_curriculum.py`), so 3- or 5-step labels both work.

Boundaries come from robot signals (gripper events + the still moment before the next action), every episode
is checked automatically (order, expected grasp/release inside each sub-task, gripper state, duration
outliers) and the review images show the camera frames at every boundary. QA result on the current data:
handover 84/84 clean, cubes 489/490 (episode 94 excluded automatically).

## Default curriculum

| Stage | Share of steps | Samples drawn from |
|---|---|---|
| 1 reach_grasp | 15% | reach_grasp (handover 50% / cubes 50%) |
| 2 transport_place | 20% | 70% new skill, 30% rehearsal of stage 1 |
| 3 bimanual | 20% | 70% bimanual (handover only), 30% earlier skills |
| 4 full_task | 45% | 70% complete instructions (both datasets), 30% all sub-tasks |

Within a stage every (dataset, skill) group gets an equal share, so the 84 handover episodes are not drowned
by the 490 cube episodes. Change it with `--curriculum plan.json`:
```json
{"stages": [
   {"name": "stage1_reach_grasp", "frac": 0.15, "skills": ["reach_grasp"]},
   {"name": "stage2_transport_place", "frac": 0.20, "skills": ["reach_grasp", "transport_place"]},
   {"name": "stage3_bimanual", "frac": 0.20, "skills": ["reach_grasp", "transport_place", "bimanual"]},
   {"name": "stage4_full_task", "frac": 0.45, "skills": ["reach_grasp", "transport_place", "bimanual", "full_task"]}],
 "new_skill_weight": 0.7}
```

## What a training sample looks like
- inputs: 3 cameras (224×224) + 16-D joint state at 6 Hz (handover: every 5th frame of 30 fps; cubes: native 6 fps)
- instruction: the sub-task text (sub-task samples) or the complete instruction (full-task samples);
  30% of the time reworded (`--paraphrase_prob`) so the model does not depend on one exact sentence
- target: next 20 actions (3.3 s). For sub-task samples the actions after the sub-task's end are **held at the
  end pose** (`--subtask_tail hold`): the policy learns to stop when the step is done, which is what a step-by-step
  orchestrator needs. `--subtask_tail mask` excludes them from the loss instead.
- cube gripper commands are shifted to the frame where the gripper actually starts moving (recording lag of
  1-4 frames; after the fix every flip is exactly one frame before the motion, like the arm actions).
  `--no_gripper_lag_fix` disables it.
- held out for validation: handover episodes in `configs/splits.json` (unseen lighting), cube episodes with index % 10 == 9

## Models and hardware
| `--policy` | Base | Notes |
|---|---|---|
| `smolvla` | `lerobot/smolvla_base` (450M, 100M trained) | ~3.5 GB at batch 8; batch 32 needs ~12 GB |
| `pi0` | `lerobot/pi0_base` (3.3B) | 24 GB+ ; add `--bf16 --override gradient_checkpointing=true` |
| `pi05` | `lerobot/pi05_base` (3.3B) | 24 GB+ ; same options; `--override train_expert_only=true` to fit smaller GPUs |

pi0 / pi0.5 use the PaliGemma tokenizer: run `huggingface-cli login` with an account that accepted the
`google/paligemma-3b-pt-224` licence. Stats (mean/std and 1st/99th percentiles for pi0.5) are computed from the
training data, so no dataset `stats.json` is used.

## Outputs (`--out`)
- `stage1_reach_grasp/ … stage4_full_task/`, `final/`: checkpoint after each stage (LeRobot format)
- `curriculum_log.json`: plan, steps per stage, sample mix, validation per dataset × (sub-task / full) at every
  `--val_every` steps and at each stage end (normalized L1 and arm error in rad)
- `norm_stats.pt`

Compare curriculum vs `--no_curriculum` on: held-out error per sub-task and for full tasks
(`evaluation/eval_offline.py`), and success / stage reached in Isaac Sim (`isaac/eval_isaac.py`). Use the per-stage checkpoints to
show how skills build up (e.g. stage-1 checkpoint given "Pick up the box with the left gripper").
