"""
scratch.py
Just a file to test out some methods
"""

import pandas as pd

import src.constants as cts
import src.preprocessing as prep

# process data, and compile into a DataFrame
df_all = prep.load_ifft_df(cts.DATADIR, filename="df_all.pkl")

# set up training and evaluation datasets
CLASSES = df_all["dronetype"].unique().tolist()
N_CLASSES = len(CLASSES)
df_all["drone_idx"] = df_all["dronetype"].map(
    pd.Series(range(N_CLASSES), index=CLASSES)
)

for x in df_all["psd"].values:
    print(x.shape)
