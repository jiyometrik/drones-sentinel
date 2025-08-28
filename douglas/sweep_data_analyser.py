import os
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from app.modules.logging.logger import logger

from app.utils.signal_io import read_bin_file

# Local constants (adjust accordingly)
SNR_THRESHOLD = -10  # dB, SNR threshold for filtering


@dataclass
class SweepConfig:
    filename: str


class SweepDataAnalyser:
    def __init__(self, config: SweepConfig):
        self.config = config
        self._validate_parameters()

        # Read sweep data from file
        self.bin_data = read_bin_file(self.config.filename, is_dict_format=True)
        self.frequencies = sorted(self.bin_data.keys())

    def _validate_parameters(self):
        validations = [
            (
                self.config.filename is not None
                and os.path.isfile(self.config.filename),
                f"File {self.config.filename} does not exist.",
            ),
        ]

        for condition, error_message in validations:
            if not condition:
                raise ValueError(error_message)

        logger.info("Configuration validated successfully")
        logger.info(f"Configuration: {self.config}")

    def compute_frequency_domain(
        self, apply_gaussian_filter=False, sigma=1.0, visualise=False
    ):
        """Convert the frequency data into a 2D array for plotting."""
        times = range(len(self.bin_data[self.frequencies[0]]))

        # Create a 2D array (rows = frequencies, columns = time samples)
        power_matrix = np.zeros((len(self.frequencies), len(times)))

        for i, freq in enumerate(self.frequencies):
            power_matrix[i, :] = self.bin_data[freq]

        # Transpose the matrix to swap axes (now rows = time, columns = frequency)
        power_matrix = power_matrix.T
        logger.debug(f"Power matrix shape: {power_matrix.shape}")

        # Apply Gaussian filter if specified
        if apply_gaussian_filter:
            if sigma <= 0:
                logger.error("Sigma must be a positive value for Gaussian filter.")
                return

            # Apply Gaussian filter to smooth the data
            power_matrix = gaussian_filter(power_matrix, sigma=sigma)

        if visualise:
            logger.debug("Plotting frequency domain data")

            # Create the frequency domain plot
            plt.figure(figsize=(12, 8))

            # Get average readings for each frequency
            avg_readings = np.mean(power_matrix, axis=0)
            print(avg_readings.shape)

            # Plot average readings
            plt.plot(self.frequencies, avg_readings)

            title = "Average Power vs Frequency"
            if apply_gaussian_filter:
                title += f" (Gaussian Filter, sigma={sigma})"
            else:
                title += " (No Gaussian Filter)"

            plt.title(title)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Average Power (dB)")
            plt.grid(True)

            plt.tight_layout()
            plt.show()

            logger.debug("Plotting spectrogram")
            # Create the spectrogram
            plt.figure(figsize=(12, 8))

            # Use imshow with transposed matrix
            # Extent defines the bounds [left, right, bottom, top]
            extent = [
                self.frequencies[0] / 1e6,
                self.frequencies[-1] / 1e6,
                0,
                len(times),
            ]  # Convert Hz to MHz

            # Use a colormap appropriate for spectrograms
            im = plt.imshow(
                power_matrix,
                aspect="auto",
                origin="lower",
                extent=extent,
                cmap="viridis",
            )

            plt.colorbar(im, label="Power (dB)")
            plt.title("RF Frequency Spectrogram")
            plt.xlabel("Frequency (MHz)")
            plt.ylabel("Time Sample")

            plt.tight_layout()
            plt.show()

        return power_matrix

    def compute(self):
        # Compute frequency domain data
        self.compute_frequency_domain(apply_gaussian_filter=False, visualise=True)
        self.compute_frequency_domain(
            apply_gaussian_filter=True, sigma=1.0, visualise=True
        )
