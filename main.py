"""
main.py
This hardcarries everything
"""

import numpy as np
import pandas as pd
import torchview
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader

import src.constants as cts
import src.mlsuite.datasets as dsets
import src.mlsuite.stft as stft
import src.model as mdl
import src.preprocessing as prep

# WARN Unload data (skip if already done)
# prep.unload_zip_files(cts.ZIPDIR, cts.DATADIR)

# Process data, and compile into a DataFrame
df_all = prep.load_ifft_df(cts.DATADIR, filename="df_all.pkl")

# Encode drone types as integers
CLASSES = df_all["dronetype"].unique().tolist()
N_CLASSES = len(CLASSES)
df_all["drone_idx"] = df_all["dronetype"].map(
    pd.Series(range(N_CLASSES), index=CLASSES)
)

# Scale STFT data
scaler, le = StandardScaler(), LabelEncoder()
X = np.stack(df_all["stft"].values)
X_scaled = scaler.fit_transform(X.reshape(X.shape[0], -1))
X = X_scaled.reshape(X.shape)
y = le.fit_transform(df_all["drone_idx"].values)

# Split data into training, validation and test sets
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.35,
    random_state=42,
    stratify=y,
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp,
)

# ...then create datasets out of them
ds_train = dsets.IFFTDataset(X_train, y_train, augment_enabled=cts.AUGMENT_TRAIN_SET)
ds_val = dsets.IFFTDataset(X_val, y_val, augment_enabled=False)
ds_test = dsets.IFFTDataset(X_test, y_test, augment_enabled=False)

dl_train = DataLoader(
    ds_train, batch_size=cts.BATCH_SIZE, shuffle=True, num_workers=cts.N_WORKERS
)
dl_val = DataLoader(
    ds_val, batch_size=cts.BATCH_SIZE, shuffle=False, num_workers=cts.N_WORKERS
)
dl_test = DataLoader(
    ds_test, batch_size=cts.BATCH_SIZE, shuffle=False, num_workers=cts.N_WORKERS
)

"""
Train any model we create in mlsuite.stft.cnn
NOTE Change model name here to try different architectures
"""
model = stft.cnn.STFTSqueezeNet(num_classes=N_CLASSES)
trainer = mdl.train_model(
    model,
    train_loader=dl_train,
    val_loader=dl_val,
    n_epochs=cts.N_EPOCHS,
)

# Then run the testing loop to get test_acc and test_f1
trainer.test(
    model=model,
    dataloaders=dl_test,
)
