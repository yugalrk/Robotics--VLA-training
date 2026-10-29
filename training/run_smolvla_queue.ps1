# Curriculum SmolVLA, low-load config: VLM + vision encoder frozen, only the action expert + projections train
# (~100M of 450M params, the standard SmolVLA fine-tune). The full fine-tune crashed the PC twice (~340 W load).
# 10k steps, effective batch 32 = 16 x 2 accumulation (moderate GPU load), target < 3 h.
# Logs: outputs/<run>/train.log (appended).
# Crash-safe: every 500 steps outputs/<run>/resume_state.pt is written; just run this script again to continue.
Set-Location (Split-Path $PSScriptRoot -Parent)  # repo root
$py = ".venv\Scripts\python.exe"
$out = "outputs/curriculum_smolvla_expert10k"
if (Test-Path "$out/final/model.safetensors") { exit }  # finished earlier
New-Item -ItemType Directory -Force $out | Out-Null
cmd /c "$py training\train_curriculum.py --policy smolvla --steps 10000 --batch 16 --grad_accum 2 --val_batches 8 --bf16 --log_every 50 --save_every 500 --resume --out $out >> $out\train.log 2>&1"
