"""Retarget the recorded left-arm motion to a moved box (FK + IK). Logic: fr3duo/retarget.py.

    python augmentation/retarget_trajectory.py --dx 0.03
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for the fr3duo package
from fr3duo.retarget import main  # noqa: E402

if __name__ == "__main__":
    main()
