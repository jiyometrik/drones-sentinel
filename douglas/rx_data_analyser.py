import os
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, stft

from app.modules.logging.logger import logger
from app.utils.signal_io import read_iq_file

# Local constants (adjust accordingly)
MAGNITUDE_THRESHOLD = 10  # Threshold for pulse pattern extraction


@dataclass
class RxConfig:
    sample_rate_hz: int  # Sample rate in Hz
    filename: str  # Filename of the IQ data file


class RxDataAnalyser:
    def __init__(self, config: RxConfig):
        self.config = config
        self._validate_parameters()

        # Read IQ data from file
        self.iq_data = read_iq_file(self.config.filename)

        # Uncomment the following if segmenting is needed
        # self.iq_data = self.iq_data[
        #     0 * (len(self.iq_data) // 5) : 1 * (len(self.iq_data) // 5)
        # ]  # Segment 1/5

    def _validate_parameters(self):
        validations = [
            (
                self.config.sample_rate_hz >= 0,
                "Sample rate must be a positive integer.",
            ),
            (
                self.config.filename is not None
                and os.path.isfile(self.config.filename),
                f"File {self.config.filename} does not exist.",
            ),
        ]

        for condition, error_message in validations:
            if not condition:
                raise ValueError(error_message)

        # logger.info("Configuration validated successfully")
        # logger.info(f"Configuration: {self.config}")

    @staticmethod
    def plot_iq(iq_data, sample_rate_hz, magnitude_only=False):
        t = np.arange(len(iq_data)) / sample_rate_hz  # Time

        if magnitude_only:
            mag = np.abs(iq_data)

            plt.figure(figsize=(12, 6))
            plt.plot(t, mag)
            plt.title("IQ Data (Magnitude Only)")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.grid(True)

            plt.show()
            return

        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(t, iq_data.real)
        plt.title("I Component (Real)")
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude")
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(t, iq_data.imag)
        plt.title("Q Component (Imaginary)")
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude")
        plt.grid(True)

        plt.show()

    @staticmethod
    def extract_pulse_pattern_from_iq(
        iq_data, sample_rate_hz, magnitude_threshold, visualise=False
    ):
        """
        Extract pulse patterns from IQ data based on a magnitude threshold.
        Anything below the threshold is reduced to 0, and everything above is set to 1.
        """
        t = np.arange(len(iq_data)) / sample_rate_hz
        mag = np.abs(iq_data)
        pulse_pattern = np.where(mag > magnitude_threshold, 1, 0)

        if visualise:
            plt.figure(figsize=(12, 6))
            plt.plot(t, pulse_pattern, label="Pulse Pattern")
            plt.title("Extracted Pulse Pattern from IQ Data")
            plt.xlabel("Time (s)")
            plt.ylabel("Pulse Pattern")
            plt.grid(True)
            plt.show()

        return pulse_pattern

    @staticmethod
    def compute_fft(iq_data, sample_rate_hz, visualise=False):
        nfft = len(iq_data)

        # Compute FFT
        fft_data = np.fft.fftshift(np.fft.fft(iq_data, nfft))

        # Frequency axis for FFT
        freq_fft = (
            np.fft.fftshift(np.fft.fftfreq(nfft, 1 / sample_rate_hz)) / 1e6
        )  # MHz

        # Find DC and Nyquist bins
        dc_bin = len(fft_data) // 2  # DC is at center after fftshift
        nyquist_bins = [0, -1]  # First and last bins are ±Nyquist

        # Set DC bin value to the average of the surrounding bins
        fft_data[dc_bin] = np.mean([fft_data[dc_bin - 1], fft_data[dc_bin + 1]])

        # Remove Nyquist bins
        fft_data = np.delete(fft_data, nyquist_bins)
        freq_fft = np.delete(freq_fft, nyquist_bins)

        # FFT magnitude spectrum (not PSD yet - this is the actual FFT magnitude)
        fft_magnitude_db = 20 * np.log10(
            np.abs(fft_data) + 1e-10
        )  # dB scale (20*log10 for magnitude)

        logger.debug(
            f"Frequency range: {freq_fft.min():.2f} to {freq_fft.max():.2f} MHz"
        )
        logger.debug(f"Sample rate value: {sample_rate_hz}")
        logger.debug(f"Sample rate type: {type(sample_rate_hz)}")

        if visualise:
            logger.debug("Plotting FFT magnitude spectrum")
            plt.figure(figsize=(12, 8))
            plt.plot(freq_fft, fft_magnitude_db)
            plt.title("FFT Magnitude Spectrum")
            plt.xlabel("Frequency Offset [MHz]")
            plt.ylabel("Magnitude [dB]")
            plt.grid(True)
            plt.xlim([freq_fft[0], freq_fft[-1]])

            plt.show()

        return fft_data, freq_fft, fft_magnitude_db

    @staticmethod
    def compute_stft(iq_data, sample_rate_hz, nperseg=8192, visualise=False):
        """
        Compute the Short-Time Fourier Transform (STFT) of the IQ data.
        """
        f, t, Zxx = stft(
            iq_data,
            fs=sample_rate_hz,
            nperseg=nperseg,
            return_onesided=False,  # Two-sided spectrum
        )

        # Shift frequencies to center around 0 Hz
        f = np.fft.fftshift(f)
        Zxx = np.fft.fftshift(Zxx, axes=0)

        if visualise:
            logger.debug("Plotting STFT")
            plt.figure(figsize=(12, 8))
            plt.pcolormesh(t, f / 1e6, np.abs(Zxx), shading="gouraud")
            plt.title("Short-Time Fourier Transform (STFT)")
            plt.xlabel("Time [s]")
            plt.ylabel("Frequency [MHz]")
            plt.colorbar(label="Magnitude")
            plt.grid(True)
            plt.show()

        return f, t, Zxx

    @staticmethod
    def compute_welch_psd(iq_data, sample_rate_hz, nperseg=8192, visualise=False):
        """
        Compute the Power Spectral Density (PSD) using Welch's method.
        """
        f, Pxx = welch(
            iq_data,
            fs=sample_rate_hz,
            nperseg=nperseg,
            return_onesided=False,  # Two-sided spectrum
        )

        # Shift frequencies to center around 0 Hz
        f = np.fft.fftshift(f)
        Pxx = np.fft.fftshift(Pxx)

        logger.debug(f"Frequency range: {f.min():.2e} to {f.max():.2e} Hz")
        logger.debug(f"Sample rate value: {sample_rate_hz}")
        logger.debug(f"Sample rate type: {type(sample_rate_hz)}")

        # Find DC (frequency = 0) and Nyquist bins
        dc_bin = len(Pxx) // 2  # DC is at center after fftshift
        nyquist_bins = [0, -1]  # First and last bins are ±Nyquist

        # Set DC bin value to the average of the surrounding bins
        Pxx[dc_bin] = np.mean([Pxx[dc_bin - 1], Pxx[dc_bin + 1]])
        # Remove Nyquist bins
        Pxx = np.delete(Pxx, nyquist_bins)
        f = np.delete(f, nyquist_bins)

        if visualise:
            logger.debug("Plotting Welch's PSD")
            plt.figure(figsize=(12, 8))
            plt.plot(f, np.sqrt(Pxx))
            plt.title("Welch's Power Spectral Density (PSD)")
            plt.xlabel("Frequency Offset [MHz]")
            plt.ylabel("PSD [V**2/Hz]")
            plt.grid(True)
            plt.xlim([f[0], f[-1]])
            plt.show()

        return f, Pxx

    def compute(self):
        RxDataAnalyser.plot_iq(
            self.iq_data, self.config.sample_rate_hz, magnitude_only=True
        )
        RxDataAnalyser.extract_pulse_pattern_from_iq(
            self.iq_data, self.config.sample_rate_hz, MAGNITUDE_THRESHOLD, True
        )

        self.compute_fft(self.iq_data, self.config.sample_rate_hz, visualise=True)
        self.compute_welch_psd(
            self.iq_data, self.config.sample_rate_hz, nperseg=8192, visualise=True
        )
