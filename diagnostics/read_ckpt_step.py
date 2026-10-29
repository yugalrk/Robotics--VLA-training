import pathlib

import numpy as np
import tensorstore as ts

CKPT = pathlib.Path("/mnt/c/Users/User/Desktop/Hack2026/C2_Teach-the-robot_VLA/trained checkpoint/fr3-duo-checkpoint-6a5f7c87-e14a-4dd1-a047-dc5659fa8327/checkpoint")
spec = {"driver": "zarr", "kvstore": {"driver": "ocdbt", "base": f"file://{CKPT / 'train_state'}", "path": "step"}}
try:
    print("training step:", int(np.asarray(ts.open(spec).result().read().result())))
except Exception as e:
    import orbax.checkpoint as ocp
    meta = ocp.PyTreeCheckpointer().metadata(CKPT / "train_state")
    print("could not read step directly:", type(e).__name__, e)
    print("metadata keys:", list(getattr(meta, "tree", meta).keys())[:10] if hasattr(getattr(meta, "tree", meta), "keys") else meta)
