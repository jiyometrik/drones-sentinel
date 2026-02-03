"""
src.preprocessing
Helpful methods to preprocess data for training and evaluation
"""

import glob
import io
import os
import zipfile
from typing import Optional

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


def compute_stft(
    iq_data: np.ndarray,
    nperseg: int = cts.TARGET_FREQ,
    noverlap: Optional[int] = cts.TARGET_TIME,
    nfft: Optional[int] = None,
    sample_rate: float = cts.SAMPLE_RATE,
    window: str = "hann",
    representation: str = "magnitude",
):
    """
    Compute STFT from IQ data.
        - iq_data: Complex IQ samples
        - nperseg: Length of each segment (window size)
        - noverlap: Number of points to overlap (default: nperseg // 2)
        - nfft: FFT size (default: nperseg, increase for zero-padding)
        - window: Window function ('hann', 'hamming', 'blackman', etc.)
        - sample_rate: Sampling frequency
    returns:
        - f: Array of frequency bins
        - t: Array of time bins
        - Zxx: STFT matrix (complex-valued)
    """

    # make IQ data complex-valued
    if not np.iscomplexobj(iq_data):
        print("[WARN] IQ data is not complex. Converting float to complex.")
        # assume interleaved I/Q if float
        if len(iq_data) % 2 == 0:
            iq_data = iq_data[::2] + 1j * iq_data[1::2]
        else:
            iq_data = iq_data.astype(np.complex64)

    f, t, Zxx = signal.stft(
        iq_data,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nperseg if nfft is None else nfft,
        window=window,
        return_onesided=False,
        boundary=None,
        padded=True,
    )

    # represent data properly
    if representation == "magnitude":
        mag, phase = np.abs(Zxx), np.angle(Zxx)
        rep = np.stack([mag, phase], axis=0)
    elif representation == "real_imag":
        rep = np.stack([Zxx.real, Zxx.imag], axis=0)
    else:
        raise ValueError(f"[ERROR] Unknown STFT representation '{representation}'")
    return f, t, rep


def pad_spectrogram(
    spectrogram: np.ndarray,
    target_freq: Optional[int] = None,
    target_time: Optional[int] = None,
    pad_mode: str = "constant",
    pad_value: float = 0.0,
) -> np.ndarray:
    """
    pad or crop spectrogram to fixed dimensions.
        - spectrogram: Input spectrogram (freq, time) or (channels, freq, time)
        - target_freq: Target frequency dimension (None = no change)
        - target_time: Target time dimension (None = no change)
        - pad_mode: Padding mode ('constant', 'edge', 'reflect', 'wrap')
        - pad_value: Value for constant padding (only used if pad_mode='constant')
    returns:
        - padded/cropped spectrogram with target dimensions
    """
    is_multichannel = spectrogram.ndim == 3
    if is_multichannel:
        _, freq_bins, time_frames = spectrogram.shape
    else:
        freq_bins, time_frames = spectrogram.shape

    # default to current dimensions if not specified
    target_freq = freq_bins if target_freq is None else target_freq
    target_time = time_frames if target_time is None else target_time

    # calculate padding/cropping for frequency axis
    freq_diff = target_freq - freq_bins
    freq_pad_before = freq_diff // 2
    freq_pad_after = freq_diff - freq_pad_before

    # calculate padding/cropping for time axis
    time_diff = target_time - time_frames
    time_pad_before = time_diff // 2
    time_pad_after = time_diff - time_pad_before

    # Handle cropping (negative padding)
    if freq_diff < 0:
        crop_start = abs(freq_pad_before)
        crop_end = freq_bins - abs(freq_pad_after)
        if is_multichannel:
            spectrogram = spectrogram[:, crop_start:crop_end, :]
        else:
            spectrogram = spectrogram[crop_start:crop_end, :]
        freq_pad_before = 0
        freq_pad_after = 0

    if time_diff < 0:
        crop_start = abs(time_pad_before)
        crop_end = time_frames - abs(time_pad_after)
        if is_multichannel:
            spectrogram = spectrogram[:, :, crop_start:crop_end]
        else:
            spectrogram = spectrogram[:, crop_start:crop_end]
        time_pad_before = 0
        time_pad_after = 0

    # Apply padding
    if (
        freq_pad_before > 0
        or freq_pad_after > 0
        or time_pad_before > 0
        or time_pad_after > 0
    ):
        if is_multichannel:
            pad_width = (
                (0, 0),  # No padding on channel axis
                (freq_pad_before, freq_pad_after),
                (time_pad_before, time_pad_after),
            )
        else:
            pad_width = (
                (freq_pad_before, freq_pad_after),
                (time_pad_before, time_pad_after),
            )

        if pad_mode == "constant":
            spectrogram = np.pad(
                spectrogram, pad_width, mode="constant", constant_values=pad_value
            )
        else:
            spectrogram = np.pad(spectrogram, pad_width, mode=pad_mode)

    return spectrogram


def visualise_wpsd(f, Pxx, output_ndarray: bool = False) -> Optional[np.ndarray]:
    """
    Visualise a Welch PSD with a power-frequency plot.
    Optionally convert graph visualisation into np.ndarray for image processing
    """
    print("[OK] Plotting wPSD")
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.semilogy(f, Pxx)

    # Remove unnecessary labels if we're generating ndarray
    ax.set_title("" if output_ndarray else "wPSD")
    ax.set_xlabel("" if output_ndarray else "Frequency (MHz)")
    ax.set_ylabel("" if output_ndarray else "Power (V^2/Hz)")
    plt.tight_layout()

    if not output_ndarray:
        plt.show()
    if output_ndarray:
        fig.canvas.draw()
        rgba_buffer = fig.canvas.buffer_rgba()
        # Returns [H, W, 4] RGBA image
        img_data = np.asarray(rgba_buffer)
        # Apply luminosity formula to turn RGBA data into grayscale (shape: [H, W])
        img_grayscale = np.dot(img_data[:, :, :3], [0.2989, 0.5870, 0.1140])
        print("[OK] Generating np.ndarray of wPSD image, size =", img_grayscale.shape)
        plt.close()
        return img_grayscale


def compute_wpsd(
    iq_data,
    sample_rate=cts.SAMPLE_RATE,
    window="hann",
    nperseg=cts.TARGET_FREQ,
    visualise=False,
):
    """
    compute_wpsd(iq_data, sample_rate, window, nperseg, visualise)
        -> (f, Pxx)
        - iq_data: numpy array of IQ data
        - sample_rate: sample rate (in Hz)
        - window: type of window to use in Welch's method
        - nperseg: number of samples per segment for Welch's method
        - visualise: whether to plot the wPSD result
    Computes the Welch power spectral density (PSD) of IFFT data
    """
    f, Pxx = signal.welch(iq_data, fs=sample_rate, nperseg=nperseg, window=window)
    print("[OK] Computed wPSD with shape =", Pxx.shape)
    if visualise:
        visualise_wpsd(f, Pxx, output_ndarray=False)
    return f, Pxx


def generate_ifft_df(datadir=cts.DATADIR) -> pd.DataFrame:
    """
    generate_ifft_df(datadir) -> pd.DataFrame
        - datadir: directory containing IFFT files
    reads all IFFT files in the specified directory and returns a DataFrame
    with columns 'dronenum' and 'ifft' containing the drone number and IFFT data
    """
    column_names = [
        "psd",
        "psd_image",
        "stft",
        "dronetype",
        "frequency_GHz",
        "distance_m",
        "gain_MHz",
    ]
    ifft_df = pd.DataFrame(columns=column_names)

    pattern = os.path.join(datadir, "**", "*.ifft")
    ifft_files = glob.glob(pattern, recursive=True)

    for fpath in ifft_files:
        # read IFFT file
        ifft_data = read_raw_ifft(fpath)

        # compute wPSD
        f, Pxx = compute_wpsd(ifft_data, visualise=False)
        psd = np.stack((f, Pxx))

        # compute np.ndarray image of wPSD
        psd_image = visualise_wpsd(f, Pxx, output_ndarray=True)

        # compute STFT
        _, _, stft = compute_stft(ifft_data, representation="magnitude")
        # pad STFT to fixed size
        stft = pad_spectrogram(
            stft,
            target_freq=cts.TARGET_FREQ,
            target_time=cts.TARGET_TIME,
        )
        print(f"[OK] STFT shape after padding = {stft.shape}")

        # grab metadata from filename
        components = os.path.relpath(fpath, datadir).split(cts.FILE_SEP)[1].split("_")
        if len(components) == 7:
            components.insert(2, "0")  # insert frequency as 0 GHz if missing
        dronetype = components[0]
        freq_ghz = int(components[2]) if int(components[2]) != 0 else 0
        distance_m = int(components[3].split("m")[0])
        gain_mhz = int(components[5])

        # and then load them into the dictionary
        entry = pd.DataFrame(
            [[psd, psd_image, stft, dronetype, freq_ghz, distance_m, gain_mhz]],
            columns=column_names,
        )
        try:
            ifft_df = pd.concat([ifft_df, entry], ignore_index=True)
        except (ValueError, TypeError) as e:
            print(f"Error processing file {fpath}: {e}. Skipping.")
    return ifft_df


def load_ifft_df(datadir=cts.DATADIR, filename="df_all.pkl") -> pd.DataFrame:
    """
    load_ifft_df(datadir, filename) -> pd.DataFrame
        - datadir: directory containing the DataFrame pickle file
        - filename: name of the pickle file
    Loads a DataFrame from a .pkl file in the specified directory
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
