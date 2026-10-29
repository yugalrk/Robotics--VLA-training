"""Training curves of a training/train_curriculum.py run as one self-contained HTML page (no extra packages).

Usage:
    python training/plot_training.py outputs/curriculum_smolvla_expert10k      # -> <run>/training_curves.html

Panels: held-out arm-joint error per dataset x instruction type (the metric that matters), training loss,
gradient norm and learning rate, with the curriculum stage boundaries. Steps logged twice (a crash and
--resume) keep the values of the resumed run.
"""
import json
import math
import re
import sys
from pathlib import Path

STEP = re.compile(r"^\[(\w+)\] step (\d+)/(\d+) loss ([\d.]+) grad ([\d.]+) lr ([\d.e+-]+)")
VAL = re.compile(r"^VAL (\d+|end of \w+): (.*)")
STAGE = re.compile(r"^=== (\w+): (\d+) steps")
VAL_ITEM = re.compile(r"([\w/]+) ([\d.]+) \(([\d.]+) rad\)")


def parse(log_path):
    train, val, stages = {}, {}, {}
    last_step = 0
    for line in Path(log_path).read_text(errors="replace").splitlines():
        if m := STEP.match(line):
            last_step = int(m[2])
            train[last_step] = {"loss": float(m[4]), "grad": float(m[5]), "lr": float(m[6]), "stage": m[1]}
        elif m := VAL.match(line):
            step = int(m[1]) if m[1].isdigit() else last_step
            val[step] = {k: math.degrees(float(rad)) for k, _, rad in VAL_ITEM.findall(m[2])}
        elif m := STAGE.match(line):
            stages.setdefault(m[1], int(m[2]))
    bounds, s = [], 0
    for name, n in stages.items():
        bounds.append({"name": name, "start": s, "end": s + n})
        s += n
    return {"train": [{"step": k, **v} for k, v in sorted(train.items())],
            "val": [{"step": k, **v} for k, v in sorted(val.items())], "stages": bounds}


PAGE = Path(__file__).with_name("plot_training_template.html")


def main():
    run = Path(sys.argv[1])
    data = parse(run / "train.log")
    data["run"] = run.name
    html = PAGE.read_text(encoding="utf-8").replace("/*DATA*/null", json.dumps(data))
    out = run / "training_curves.html"
    out.write_text(html, encoding="utf-8")
    print(f"{len(data['train'])} train points, {len(data['val'])} validations, {len(data['stages'])} stages -> {out}")


if __name__ == "__main__":
    main()
