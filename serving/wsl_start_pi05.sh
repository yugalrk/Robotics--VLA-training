#!/bin/bash
# Run the pi0.5 policy server in the foreground (keep the wsl.exe that runs this open). Log: eval_results/pi05_server.log
P=/mnt/c/Users/User/Desktop/Hack2026/C2_Teach-the-robot_VLA
cd ~/openpi
pkill -f fr3duo_pi05_server.py 2>/dev/null
export XLA_PYTHON_CLIENT_PREALLOCATE=false
exec .venv/bin/python -u "$P/serving/fr3duo_pi05_server.py" \
  --checkpoint "$P/trained checkpoint/fr3-duo-checkpoint-6a5f7c87-e14a-4dd1-a047-dc5659fa8327/checkpoint" \
  > "$P/eval_results/pi05_server.log" 2>&1 < /dev/null
