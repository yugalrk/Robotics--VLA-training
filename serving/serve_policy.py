"""Serve a LeRobot checkpoint over TCP (used by isaac/eval_isaac.py). Logic: fr3duo/policy_runner.py.

    python serving/serve_policy.py --checkpoint outputs/smolvla_handover/step_003000 --port 5555 [--selftest]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for the fr3duo package
from fr3duo.policy_runner import main  # noqa: E402

if __name__ == "__main__":
    main()
