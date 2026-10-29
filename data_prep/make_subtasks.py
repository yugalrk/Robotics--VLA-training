"""Per-frame sub-task labels + quality checks for both datasets. Logic: fr3duo/subtasks.py.

    python data_prep/make_subtasks.py [--only handover|cubes] [--handover_subtasks 3|5] [--no_write]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for the fr3duo package
from fr3duo.subtasks import main  # noqa: E402

if __name__ == "__main__":
    main()
