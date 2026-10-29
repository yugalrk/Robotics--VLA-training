"""Write configs/splits.json (10 held-out lighting episodes) and check LeRobot can load a sample.

    python data_prep/make_splits.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for the fr3duo package
from fr3duo.data import main  # noqa: E402

if __name__ == "__main__":
    main()
