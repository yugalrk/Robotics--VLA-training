"""Write geometry/head_camera.json from the scene USD (needs the USD libs). Logic: fr3duo/head_camera.py.

    python augmentation/head_camera_model.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for the fr3duo package
from fr3duo.head_camera import main  # noqa: E402

if __name__ == "__main__":
    main()
