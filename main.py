"""
main.py
this hardcarries everything
"""

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader

import src.constants as cts
import src.mlsuite.cnn as cnn
import src.mlsuite.datasets as dsets
import src.model as mdl
import src.preprocessing as prep

# unload data [DONE, skip if already done]
# prep.unload_zip_files(cts.ZIPDIR, cts.DATADIR)

# process data, and compile into a DataFrame
df_all = prep.load_ifft_df(cts.DATADIR, filename="df_all.pkl")

# set up training and evaluation datasets
CLASSES = df_all["dronetype"].unique().tolist()
N_CLASSES = len(CLASSES)
df_all["drone_idx"] = df_all["dronetype"].map(
    pd.Series(range(N_CLASSES), index=CLASSES)
)

"""
creates training and validation datasets, where
* X are the wPSDs, and 
* y are the drone types, indexed 0--3
"""
scaler, le = StandardScaler(), LabelEncoder()
X = np.stack(df_all["psd"].values)
X_scaled = scaler.fit_transform(X.reshape(X.shape[0], -1))
X = X_scaled.reshape(X.shape)
y = le.fit_transform(df_all["drone_idx"].values)

# split dataframe into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ...then create datasets out of them
ds_train = dsets.PSDDataset(X_train, y_train)
ds_test = dsets.PSDDataset(X_test, y_test)
dl_train = DataLoader(ds_train, batch_size=64, shuffle=True, num_workers=0)
dl_test = DataLoader(ds_test, batch_size=64, shuffle=False, num_workers=0)

"""
train any models we create in src/cnn.py with one-shot training loop in src/model.py
"""
# NOTE change model name here to try different architectures
model = cnn.VGG_PSDClassifier(num_classes=N_CLASSES, lr=cts.LR)
print(model)
trainer = mdl.train_model(
    model,
    train_loader=dl_train,
    val_loader=dl_test,
    n_epochs=cts.N_EPOCHS,
)
