"""Quick look at the raw handover data: episode lengths, gripper range, action vs next state, success labels.

    python data_prep/quick_look.py
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd

DS = Path(__file__).resolve().parents[1] / "Datasets/FR3Duo_Handover_Dataset_lerobot3"

df = pd.concat(pd.read_parquet(f) for f in glob.glob(str(DS / "data/chunk-000/file-000.parquet")))
S, A = np.stack(df["observation.state"]), np.stack(df["action"])

print(df.groupby("episode_index").size().describe())        # episode lengths
print("gripper range:", A[:, [7, 15]].min(0), A[:, [7, 15]].max(0))
# absolute targets? action[t] should be close to state[t+1]
same_ep = df["episode_index"].values[:-1] == df["episode_index"].values[1:]
print("|a_t - s_t+1|:", np.abs(A[:-1][same_ep] - S[1:][same_ep]).mean(0).round(4))
print("|a_t - s_t|  :", np.abs(A - S).mean(0).round(4))
print(df.groupby("episode_index")["success"].agg(["mean", "last"]))  # success labelling
print(pd.read_parquet(DS / "meta/tasks.parquet"))  # task instruction
