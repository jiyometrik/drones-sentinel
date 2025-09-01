"""
dataloader.py
A general script for extracting the raw collected data from the zip files and organising them
"""

import glob
import os
import zipfile

import numpy as np
import pandas as pd
import tsfel

DATADIR = os.path.join(os.getcwd(), "data")
ZIPDIR = os.path.join(os.getcwd(), "data/zip")


def unload_zip_files(zipdir, extractdir) -> None:
    """unload zip files in the respective zip directory"""
    extension = ".zip"

    # list filenames that end with .zip
    zfile_matches = []
    for fname in os.listdir(zipdir):
        if fname.endswith(extension) and os.path.isfile(os.path.join(zipdir, fname)):
            zfile_matches.append(os.path.join(zipdir, fname))
    print(zfile_matches)

    # unzip all zipfiles found
    for zfile in zfile_matches:
        try:
            with zipfile.ZipFile(zfile, "r") as zip_ref:
                zip_ref.extractall(extractdir)
            print(
                f"Successfully extracted all files from '{zipdir}' to '{extractdir}'."
            )
        except zipfile.BadZipFile:
            print(f"Error: '{zfile}' is not a valid ZIP file.")
        except FileNotFoundError:
            print(f"Error: The file '{zfile}' was not found.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


def load_ifft(path):
    """helper function to load all IFFT data. to be used in conjunction with `get_features`."""
    test_dtypes = [np.float32, np.float64, np.complex64, np.complex128]
    for dtype in test_dtypes:
        try:
            data = np.fromfile(path, dtype=dtype)
            if data.size > 0:
                return data, dtype
        except:
            continue
    raise ValueError(f"Could not interpret {path}")


def get_features(datadir):
    """
    obtain the tsfel features from IFFT files, according to a specified data directory
    """
    cfg = tsfel.get_features_by_domain()
    ifft_fpaths = glob.glob(os.path.join(datadir, "*", "*", "sweep", "*.ifft"))

    feature_list, labels = [], []
    fpaths_clean = []
    for ifft_fpath in ifft_fpaths:
        try:
            data, _ = load_ifft(ifft_fpath)
            signal = np.abs(data) if np.iscomplexobj(data) else data

            # Extract TSFEL features
            feature_df = tsfel.time_series_features_extractor(cfg, signal, fs=1000)
            feature_list.append(feature_df)

            # Correct label: parent folder of 'sweep'
            label = os.path.basename(os.path.dirname(os.path.dirname(ifft_fpath)))
            labels.append(label)
            fpaths_clean.append(ifft_fpath)

        except Exception as e:
            print(f"Skipping {ifft_fpath}: {e}")

    X = pd.concat(feature_list, ignore_index=True)
    y = pd.Series(labels, name="label")
    return X, y


if __name__ == "__main__":
    unload_zip_files(zipdir=ZIPDIR, extractdir=DATADIR)
    get_features(datadir=DATADIR)
