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

# DRONETYPES = ['Aquila16', 'Background', 'DjiMini2', 'DjiRcN1']
LR = 1e-3
N_EPOCHS = 20

# dimensions for PSDs and STFTs
TARGET_FREQ = 512
TARGET_TIME = 256
