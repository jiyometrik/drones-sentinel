"""
src.constants
A list of helpful constants (filepaths, sample rates, etc.) used throughout the project
"""

import os
import sys

OPERATING_SYSTEM = sys.platform
FILE_SEP = "\\" if OPERATING_SYSTEM == "win32" else "/"

DATADIR = os.path.join(os.getcwd(), "data")
ZIPDIR = os.path.join(os.getcwd(), "data/zip")
SAMPLE_RATE = 100_000_000

LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 16
N_EPOCHS = 20
AUGMENT_TRAIN_SET = False
N_WORKERS = 0 if OPERATING_SYSTEM == "win32" else 4

# dimensions for PSDs and STFTs
TARGET_FREQ = 512
TARGET_TIME = 256
