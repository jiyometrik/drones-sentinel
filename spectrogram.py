"""
spectrogram.py
helper file to generate 2d spectrograms from 1d time series IFFT data
"""

import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

import sourdough as sd

SPECTRODIR = os.path.join(os.getcwd(), "data/spectrograms")


def create_spectrogram(
    signal_data,
    fs=1000,
    nperseg=256,
    noverlap=128,
    output_size=(224, 224),
    method="stft",
):
    """
    Create a spectrogram from 1D signal data
    """

    if method == "stft":
        # Short-Time Fourier Transform
        frequencies, times, Sxx = signal.spectrogram(
            signal_data, fs=fs, nperseg=nperseg, noverlap=noverlap
        )
        # Convert to dB scale
        Sxx_db = 10 * np.log10(Sxx + 1e-10)  # Add small value to avoid log(0)

    elif method == "cwt":
        # Continuous Wavelet Transform using Morlet wavelet
        widths = np.arange(1, 31)  # Scale parameter for wavelets
        Sxx_db = signal.cwt(signal_data, signal.morlet2, widths)
        Sxx_db = np.abs(Sxx_db)
        Sxx_db = 20 * np.log10(Sxx_db + 1e-10)

    # Normalize to 0-255 range
    Sxx_normalized = (
        (Sxx_db - Sxx_db.min()) / (Sxx_db.max() - Sxx_db.min()) * 255
    ).astype(np.uint8)

    # Resize to desired output size
    spectrogram_resized = cv2.resize(Sxx_normalized, output_size)

    return spectrogram_resized


def save_spectrogram_image(spectrogram, filepath, colormap="viridis"):
    """
    Save spectrogram as an image file
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(spectrogram, aspect="auto", origin="lower", cmap=colormap)
    plt.axis("off")  # Remove axes for clean image
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches="tight", pad_inches=0, dpi=150)
    plt.close()


def get_spectrograms(
    datadir,
    spectro_dir,
    fs=1000,
    image_size=(224, 224),
    save_images=True,
    method="stft",
):
    """
    Generate spectrograms from IFFT files and optionally save them as images
    """

    # Create spectrograms directory if it doesn't exist
    if save_images:
        os.makedirs(spectro_dir, exist_ok=True)

    ifft_fpaths = glob.glob(os.path.join(datadir, "*", "*", "sweep", "*.ifft"))

    spectrograms = []
    labels = []
    fpaths_clean = []

    for ifft_fpath in ifft_fpaths:
        try:
            data, _ = sd.load_ifft(ifft_fpath)
            signal_data = np.abs(data) if np.iscomplexobj(data) else data

            # Generate spectrogram
            spectrogram = create_spectrogram(
                signal_data, fs=fs, output_size=image_size, method=method
            )

            # Get label from directory structure
            label = os.path.basename(os.path.dirname(os.path.dirname(ifft_fpath)))

            spectrograms.append(spectrogram)
            labels.append(label)
            fpaths_clean.append(ifft_fpath)

            # Save spectrogram as image if requested
            if save_images:
                # Create subdirectory for each label
                label_dir = os.path.join(spectro_dir, label)
                os.makedirs(label_dir, exist_ok=True)

                # Generate filename
                base_name = os.path.basename(ifft_fpath).replace(".ifft", ".png")
                image_path = os.path.join(label_dir, base_name)

                # Save as both grayscale numpy array and colored image
                save_spectrogram_image(spectrogram, image_path)

            print(f"Processed: {ifft_fpath}")

        except Exception as e:
            print(f"Skipping {ifft_fpath}: {e}")

    # Convert to numpy arrays
    spectrograms = np.array(spectrograms)

    # Add channel dimension for grayscale (for CNN compatibility)
    if len(spectrograms.shape) == 3:
        spectrograms = spectrograms[..., np.newaxis]

    # Convert to RGB format if needed (duplicate grayscale across 3 channels)
    # spectrograms = np.repeat(spectrograms, 3, axis=-1)

    labels = pd.Series(labels, name="label")

    print(f"Generated {len(spectrograms)} spectrograms of shape {spectrograms.shape}")
    print(f"Labels distribution:\n{labels.value_counts()}")

    return spectrograms, labels
