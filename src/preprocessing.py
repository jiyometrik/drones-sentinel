"""
src.preprocessing
helpful methods to preprocess data for training and evaluation
"""

import glob
import os
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from tqdm import tqdm

import src.constants as cts


def unload_zip_files(zipdir=cts.ZIPDIR, extractdir=cts.DATADIR) -> None:
    """
    unload_zip_files(zipdir, extractdir) -> None
        - zipdir: directory containing zip files to be extracted
        - extractdir: directory to extract zip files in
    unloads zip files in the respective zip directory
    """

    # list filenames that end with .zip
    pattern = os.path.join(zipdir, "**", "*.zip")
    zipfiles = glob.glob(pattern, recursive=True)

    # unzip all zipfiles found
    for zfile in tqdm(zipfiles):
        try:
            with zipfile.ZipFile(zfile, "r") as zip_ref:
                zip_ref.extractall(extractdir)
            print(
                f"[OK] Successfully extracted all files from '{zipdir}' to '{extractdir}'"
            )
        except zipfile.BadZipFile:
            print(f"[XXX] '{os.path.relpath(zfile, zipdir)}' is an invalid .zip file")
        except FileNotFoundError:
            print(f"[XXX] File '{os.path.relpath(zfile, zipdir)}' was not found")
    return None


def read_raw_ifft(fpath, datadir=cts.DATADIR) -> np.ndarray:
    """
    read_raw_ifft(fpath) -> np.ndarray
        - fpath: path to the raw binary file
        - datadir: directory containing IFFT files
    read raw binary data from file and return as numpy array after IFFT.
    """
    # test different data types
    dtypes_to_try = [np.float32, np.float64, np.complex64, np.complex128]
    for dt in dtypes_to_try:
        try:
            data = np.fromfile(fpath, dtype=dt)
            if data.size > 0:
                print(
                    f"[OK] {os.path.relpath(fpath, datadir)} loaded with dtype={dt}, shape={data.shape}"
                )
                return data
        except Exception as e:
            print(f"[XXX] Failed {dt} for {os.path.relpath(fpath, datadir)}: {e}")
    raise ValueError(
        f"Could not interpret {os.path.relpath(fpath, datadir)} with common dtypes"
    )


def compute_stft(iq_data, sample_rate=cts.SAMPLE_RATE, nperseg=8192, visualise=False):
    """
    compute_stft(iq_data, sample_rate, nperseg=8192, visualise=False) -> (f, t, Zxx)
        - iq_data: numpy array of IQ data
        - sample_rate: sample rate (in Hz)
        - nperseg: number of samples per segment for STFT
        - visualise: whether to plot the STFT result
    computes the short-time fourier transform of IFFT data, in numpy.ndarray
    """
    f, t, Zxx = signal.stft(
        iq_data,
        fs=sample_rate,
        nperseg=nperseg,
        return_onesided=False,  # two-sided spectrum
    )

    # Shift frequencies to center around 0 Hz
    f = np.fft.fftshift(f)
    Zxx = np.fft.fftshift(Zxx, axes=0)

    if visualise:
        print("[OK] Plotting STFT")
        plt.figure(figsize=(12, 8))
        plt.pcolormesh(t, f / 1e6, np.abs(Zxx), shading="gouraud")
        plt.title("Short-Time Fourier Transform (STFT)")
        plt.xlabel("Time [s]")
        plt.ylabel("Frequency [MHz]")
        plt.colorbar(label="Magnitude")
        plt.grid(True)
        plt.show()

    return f, t, Zxx


def compute_welch_psd(
    iq_data, sample_rate=cts.SAMPLE_RATE, window="hann", nperseg=16_384, visualise=False
):
    """
    compute_welch_psd(iq_data, sample_rate, window='hann', nperseg=8192, visualise=False)
        -> (f, Pxx)
        - iq_data: numpy array of IQ data
        - sample_rate: sample rate (in Hz)
        - window: type of window to use in Welch's method
        - nperseg: number of samples per segment for Welch's method
        - visualise: whether to plot the wPSD result
    computes the Welch power spectral density (PSD) of IFFT data
    """
    f, Pxx = signal.welch(iq_data, fs=sample_rate, nperseg=nperseg, window=window)

    if visualise:
        print("[OK] Plotting wPSD")
        plt.figure(figsize=(12, 8))
        plt.semilogy(f, Pxx)
        plt.title("wPSD")
        plt.xlabel("Frequency [MHz]")
        plt.ylabel("Power [V^2/Hz]")
        plt.show()

    return f, Pxx


def generate_ifft_df(datadir=cts.DATADIR) -> pd.DataFrame:
    """
    generate_ifft_df(datadir) -> pd.DataFrame
        - datadir: directory containing IFFT files
    reads all IFFT files in the specified directory and returns a DataFrame
    with columns 'dronenum' and 'ifft' containing the drone number and IFFT data
    """
    ifft_df = pd.DataFrame(
        columns=["psd", "stft", "dronetype", "frequency_ghz", "distance_m", "gain_mhz"]
    )

    pattern = os.path.join(datadir, "**", "*.ifft")
    ifft_files = glob.glob(pattern, recursive=True)

    for fpath in ifft_files:
        # read IFFT file
        ifft_data = read_raw_ifft(fpath)

        # compute wPSD
        f, Pxx = compute_welch_psd(ifft_data, visualise=False)
        psd = np.stack((f, Pxx))

        # compute STFT
        _, _, stft = compute_stft(ifft_data, visualise=False)

        # grab the metadata in the filename
        components = os.path.relpath(fpath, datadir).split("\\")[1].split("_")
        if len(components) == 7:
            components.insert(2, "0")  # insert frequency as 0 GHz if missing
        dronetype = components[0]
        freq_ghz = int(components[2]) if int(components[2]) != 0 else 0
        distance_m = int(components[3].split("m")[0])
        gain_mhz = int(components[5])

        # and then load them into the dictionary
        record = {
            "psd": psd,
            "stft": stft,
            "dronetype": dronetype,
            "frequency_ghz": freq_ghz,
            "distance_m": distance_m,
            "gain_mhz": gain_mhz,
        }
        try:
            ifft_df = pd.concat(
                [ifft_df, pd.Series(record).to_frame().T], ignore_index=True
            )
        except (ValueError, TypeError) as e:
            print(f"Error processing file {fpath}: {e}. Skipping.")
    return ifft_df


def load_ifft_df(datadir=cts.DATADIR, filename="df_all.pkl") -> pd.DataFrame:
    """
    load_ifft_df(datadir, filename) -> pd.DataFrame
        - datadir: directory containing the DataFrame pickle file
        - filename: name of the pickle file
    loads a DataFrame from a pickle file in the specified directory
    """
    fpath = os.path.join(datadir, filename)
    if os.path.exists(fpath):
        df = pd.read_pickle(fpath)
        print(f"[OK] Loaded DataFrame from '{fpath}' with shape {df.shape}")
        return df
    else:
        df = generate_ifft_df(datadir)
        df.to_pickle(fpath)
        return df
