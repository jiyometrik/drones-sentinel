from typing import Any, Dict, List, Tuple

import numpy as np
import tensorflow as tf
import tsfel
from scipy.interpolate import interp1d

from app.modules.logging.logger import logger
from app.utils.psd import welch_tf
from app.utils.signal_io import read_iq_file, parse_iq_data


class FeatureExtractor:
    """Handles feature extraction from IQ data."""

    @staticmethod
    def load_iq_data(iq_file_path: str) -> np.ndarray:
        """Load IQ data from a file."""
        logger.info(f"Loading IQ data from {iq_file_path}")
        iq_data = read_iq_file(iq_file_path)
        return parse_iq_data(iq_data)

    @staticmethod
    def interpolate_to_length(data: np.ndarray, target_length: int) -> np.ndarray:
        """Interpolate data to match target length."""
        if len(data) == target_length:
            return data

        interp_func = interp1d(
            np.linspace(0, 1, len(data)),
            data,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        return interp_func(np.linspace(0, 1, target_length))

    @staticmethod
    def normalize_data(data: np.ndarray) -> np.ndarray:
        """Normalize data using z-score normalization."""
        data = data.copy()  # Avoid modifying original data

        if data.ndim == 2 and data.shape[1] >= 2:  # I/Q pairs
            # Normalize I and Q separately
            for i in range(min(2, data.shape[1])):
                channel = data[:, i]
                mean_val = np.mean(channel)
                std_val = np.std(channel)
                if std_val > 0:
                    data[:, i] = (channel - mean_val) / std_val
                else:
                    data[:, i] = channel - mean_val
        else:
            # Single channel normalization
            mean_val = np.mean(data)
            std_val = np.std(data)
            if std_val > 0:
                data = (data - mean_val) / std_val
            else:
                data = data - mean_val

        return data

    @staticmethod
    def extract_custom_fft_features(
        complex_signal: np.ndarray,
        sample_rate_hz: float,
        nfft: int,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        if not np.iscomplexobj(complex_signal):
            raise ValueError("Input signal must be a complex numpy array.")

        # Compute FFT using TensorFlow
        tf_signal = tf.convert_to_tensor(complex_signal, dtype=tf.complex64)
        fft_data = tf.signal.fft(tf_signal)

        fft_magnitude_db = 20 * np.log10(
            tf.abs(fft_data).numpy() + 1e-10
        )  # Avoid log(0)

        # Remove DC spike by setting the first element to first non-DC value
        if len(fft_magnitude_db) > 1:
            fft_magnitude_db[0] = fft_magnitude_db[1]

        # Compute Welch PSD
        _, Pxx = welch_tf(
            complex_signal,
            fs=sample_rate_hz,
            nperseg=min(1024, nfft // 4),
            return_onesided=False,  # Two-sided spectrum
        )

        # Convert to dB scale
        Pxx_db = 10 * np.log10(Pxx + 1e-10)  # Avoid log(0)

        return fft_magnitude_db, Pxx_db

    @staticmethod
    def extract_tsfel_features(
        data: np.ndarray, sample_rate_hz: float, tsfel_config: Dict
    ) -> np.ndarray:
        """
        Extract TSFEL features from the data.

        Args:
            data: Input data (expects 2D array with I/Q channels)
            sample_rate_hz: Sampling rate
            tsfel_config:
                TSFEL configuration dictionary taken
                from tsfel.get_features_by_domain()

        Returns:
            Extracted features as a 2D numpy array.
        """
        # Convert I/Q data to format expected by TSFEL
        if data.ndim == 2 and data.shape[1] >= 2:
            # For I/Q data, extract features from magnitude
            complex_signal = data[:, 0] + 1j * data[:, 1]
            magnitude = np.abs(complex_signal)

            # TSFEL expects (n_samples, 1) for univariate time series
            tsfel_input = magnitude.reshape(-1, 1)
        else:
            tsfel_input = data.reshape(-1, 1)

        # Extract features
        features = tsfel.time_series_features_extractor(
            tsfel_config, tsfel_input, fs=sample_rate_hz, verbose=0
        )

        # Return flattened feature vector
        return features.values.flatten()

    @staticmethod
    def extract_combined_features(
        data: np.ndarray,
        sample_rate_hz: float,
        use_custom_fft: bool = True,
        use_tsfel: bool = False,
        tsfel_feature_domains: List[str] = None,
    ) -> Tuple[Dict[str, Any], np.ndarray]:
        """Extract features using both custom RF-specific and TSFEL features.

        Args:
            data: Input IQ data
            sample_rate_hz: Sampling rate
            use_custom_fft: Whether to include custom FFT features
            use_tsfel: Whether to include TSFEL features
            tsfel_feature_domains: List of TSFEL feature domains to include

        Returns:
            Tuple containing:
            - features_metadata: Dictionary with metadata about features
            - features_array: Numpy array of extracted features

        TODO: Fix TSFEL spectogram returning one-sided spectrum only
        """
        features_metadata = {
            "input_info": {
                "data_shape": data.shape,
                "sample_rate_hz": sample_rate_hz,
                "data_length": len(data),
            },
            "feature_names": [],
            "total_features": 0,
        }
        features_list = []
        feature_names = []

        # Add original I/Q data
        if data.ndim == 2 and data.shape[1] >= 2:
            i = data[:, 0]
            q = data[:, 1]

            features_list.extend([i, q])  # I, Q channels

            # Add to metadata
            feature_names.extend(["I", "Q"])
        else:
            features_list.append(data.flatten())

            # Add to metadata
            feature_names.append("IQ")

        # Add custom RF-specific features
        if use_custom_fft:
            if data.ndim == 2 and data.shape[1] >= 2:
                complex_signal = data[:, 0] + 1j * data[:, 1]
            else:
                complex_signal = data.flatten() if data.ndim > 1 else data

            nfft = len(complex_signal)
            fft_magnitude_db, Pxx_db = FeatureExtractor.extract_custom_fft_features(
                complex_signal, sample_rate_hz, nfft
            )

            # Interpolate PSD to match data length
            Pxx_interpolated = FeatureExtractor.interpolate_to_length(Pxx_db, len(data))

            # Combine custom features
            features_list.extend([fft_magnitude_db, Pxx_interpolated])

            # Add to metadata
            feature_names.extend(["FFT_Magnitude_dB", "PSD_dB"])

        # Add TSFEL features
        if use_tsfel:
            if tsfel_feature_domains is None:
                tsfel_feature_domains = ["statistical", "temporal", "spectral"]

            # Get TSFEL configuration
            cfg = tsfel.get_features_by_domain(domain=tsfel_feature_domains)

            # Extract TSFEL features
            tsfel_features = FeatureExtractor.extract_tsfel_features(
                data, sample_rate_hz, cfg
            )

            # Get the actual feature names from TSFEL extraction
            if data.ndim == 2 and data.shape[1] >= 2:
                complex_signal = data[:, 0] + 1j * data[:, 1]
                magnitude = np.abs(complex_signal)
                tsfel_input = magnitude.reshape(-1, 1)
            else:
                tsfel_input = data.reshape(-1, 1)

            # Extract features to get the actual feature dataframe
            features_df = tsfel.time_series_features_extractor(
                cfg, tsfel_input, fs=sample_rate_hz, verbose=0
            )

            # Get actual column names with proper multi-value feature handling
            actual_tsfel_names = [
                f"tsfel_{col} (Broadcasted)" for col in features_df.columns
            ]

            # Add TSFEL features to list (broadcasted to match data length)
            for i, feature_val in enumerate(tsfel_features):
                features_list.append(np.full(len(data), feature_val))

            # Add to metadata using actual feature names
            feature_names.extend(actual_tsfel_names)

        # Stack all features
        stacked_features = np.stack(features_list, axis=-1)

        # Update metadata with actual counts
        features_metadata["feature_names"] = feature_names
        features_metadata["total_features"] = len(feature_names)
        features_metadata["actual_feature_shape"] = stacked_features.shape

        logger.debug(
            f"Extracted {features_metadata['total_features']} "
            f"features: {features_metadata['feature_names']}"
        )

        return features_metadata, stacked_features
