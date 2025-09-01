"""
testrun.py
The main script for running the helper files.
To be used to test out difference classification model architectures.
"""

import dataloader as dl
from mlsuite import random_forest

# Preprocess all IFFT files
dl.unload_zip_files(zipdir=dl.ZIPDIR, extractdir=dl.DATADIR)
X, y = dl.get_features(datadir=dl.DATADIR)

# Get top features using random forest
rf = random_forest.RandomForest(X, y)
rf.train()
rf.plot_n_features()
