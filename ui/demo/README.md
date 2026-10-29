# Demo UIs

Static explainer pages. Each `build_*` / `make_*` script reads the datasets (not in git) and writes a folder of
static files; `serve_static.py` serves a folder with HTTP Range support so the videos can seek.
Run everything from the repo root with the project venv.

| Page | Build | Serve (example port) |
|---|---|---|
| Robot handover explorer (3 pages: data, robot model, camera → grid → LUT) | `python ui/demo/build_site.py` → `build/site/` | `python ui/demo/serve_static.py --dir build/site --port 8779` |
| Sub-task splitting (how `data_prep/make_subtasks.py` cuts an episode) | `python ui/demo/build_subtask_demo.py [--episode 0]` → `build/subtask_demo/` | `python ui/demo/serve_static.py --dir build/subtask_demo --port 8780` |
| Box grid player (LUT of retargeted trajectories) | `python augmentation/augment_box_grid.py`, then `python augmentation/camera_grid.py` → `Datasets/fr3_duo_handover_84_ep/Augmented/ui/` | `python ui/demo/serve_static.py --dir Datasets/fr3_duo_handover_84_ep/Augmented --port 8778` (open `/ui/`) |
| Video-synced gripper XYZ | `python ui/demo/make_xyz_verify.py` | serve the folder it prints |
| Gripper path over a time window | `python ui/demo/plot_tcp_window.py [--start 7.6 --seconds 3.33]` | single self-contained HTML file |
| Box-shift retarget viewer | `python augmentation/retarget_trajectory.py --dx 0.03` (uses `retarget_template.html`) | serve the folder it prints |

Files

- `*_template.html`, `site/*.html`: page templates copied next to the generated data.
- `vendor/plotly-2.35.2.min.js`: Plotly (MIT), copied into each page so the pages work offline.
- The robot model behind these pages: `fr3duo/kinematics.py` + `geometry/fr3duo_scene_chain.json` (from the scene USD via
  `augmentation/extract_scene_kinematics.py`); it matches Franka's official FR3v2 / FR3 Duo description
  (github.com/frankarobotics/franka_description) to 0.000 mm.
