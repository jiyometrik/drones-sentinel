import struct

import numpy as np

from app.modules.logging.logger import logger


def read_iq_file(filename, dtype=np.int8):
    # Read raw bytes
    raw_data = np.fromfile(filename, dtype=dtype)

    # HackRF hackrf_transfer saves interleaved I and Q samples
    # Convert to complex numbers
    iq_data = raw_data[0::2] + 1j * raw_data[1::2]

    logger.debug(f"Read {len(iq_data)} IQ samples from {filename}")

    return iq_data


def parse_iq_data(iq_data):
    # Split into real and imaginary parts
    real = np.real(iq_data)
    imag = np.imag(iq_data)
    stack = np.stack([real, imag], axis=-1)

    # Stack and return as a tensor
    logger.debug(f"IQ data shape: {stack.shape}")
    return stack


def detect_hackrf_sweep_format(filename):
    """
    Detect if file is binary (-B) or IFFT (-I) format.
    Returns:
        "binary" if -B format, "ifft" if -I format, None if unknown.
    """
    with open(filename, "rb") as f:
        # Read first few bytes to analyze structure
        first_bytes = f.read(16)
        if len(first_bytes) < 16:
            logger.error(f"File {filename} is too short to determine format.")
            return None

        f.seek(0)  # Reset file pointer to the start

        try:
            # Try to parse as binary format first
            record_length = struct.unpack("<I", first_bytes[:4])[0]

            # Binary format validation checks:
            # 1. Record length should be reasonable (16 + power_values*4)
            # 2. Should be divisible correctly
            if record_length >= 16 and (record_length - 16) % 4 == 0:
                # Try to read the band edges
                band_edge_low = struct.unpack("<Q", first_bytes[4:12])[0]
                band_edge_high = struct.unpack("<Q", first_bytes[12:20])[0]

                # Sanity checks for frequency values
                if (
                    1000000 <= band_edge_low <= 6000000000  # 1MHz to 6GHz
                    and band_edge_low < band_edge_high
                    and (band_edge_high - band_edge_low) <= 100000000
                ):  # Max 100MHz span
                    return "binary"

            # If binary parsing failed, try IFFT format
            # IFFT format is just alternating float32 I/Q pairs
            f.seek(0)
            test_data = f.read(32)  # Read 8 float32 values (4 I/Q pairs)

            if len(test_data) == 32:
                # Try to unpack as floats
                floats = struct.unpack("<8f", test_data)

                # Basic sanity check: values should be reasonable for I/Q data
                # (not too large, not all zeros, etc.)
                if all(abs(val) < 100.0 for val in floats) and any(
                    val != 0 for val in floats
                ):
                    return "ifft"

        except (struct.error, ValueError):
            pass

    logger.error(f"Could not determine format for {filename}")
    return None


def read_bin_file(filename, is_dict_format=False):
    is_eof = False
    data_list = []

    with open(filename, "rb") as f:
        while not is_eof:
            try:
                # Try to read record_length (4 bytes)
                record_length_bytes = f.read(4)
                if not record_length_bytes or len(record_length_bytes) < 4:
                    is_eof = True
                    break

                # Unpack record length
                record_length = struct.unpack("<I", record_length_bytes)[0]

                # Calculate number of power values
                # record_length = 2*sizeof(band_edge) + num_samples*sizeof(float)
                # where sizeof(band_edge)=8 and sizeof(float)=4
                num_power_values = (record_length - 16) // 4

                # Read band_edge frequencies (2 uint64 values)
                band_edge_low = struct.unpack("<Q", f.read(8))[0]
                band_edge_high = struct.unpack("<Q", f.read(8))[0]

                # Read power values (array of floats)
                power_format = f"<{num_power_values}f"
                power_bytes = f.read(num_power_values * 4)

                if len(power_bytes) < num_power_values * 4:
                    print(f"Warning: Incomplete record at position {f.tell()}")
                    break

                sweep_data = struct.unpack(power_format, power_bytes)
                step = (band_edge_high - band_edge_low) / len(sweep_data)
                x_axis = np.arange(band_edge_low + step / 2, band_edge_high, step)

                sweep_data = {
                    "x": x_axis.tolist(),  # Frequencies in Hz
                    "y": list(sweep_data),  # Power levels in dBm
                }

                data_list.append(sweep_data)

            except Exception as e:
                logger.error(
                    f"Error reading record at position {f.tell()}: "
                    f"{e}. Skipping record..."
                )
                continue

    logger.info(f"Read {len(data_list)} records from {filename}")

    if is_dict_format:
        data_dict = {}

        for record in data_list:
            x = record["x"]
            y = record["y"]

            for freq, reading in zip(x, y):
                if freq not in data_dict:
                    data_dict[freq] = []
                data_dict[freq].append(reading)

        # If shape of data_dict is not consistent, trim last values
        min_length = min(len(v) for v in data_dict.values())

        for freq in data_dict:
            data_dict[freq] = data_dict[freq][:min_length]

        return data_dict

    return data_list


def read_ifft_file(filename):
    with open(filename, "rb") as f:
        # Read entire file as float32 pairs
        file_data = f.read()

        num_floats = len(file_data) // 4
        if num_floats % 2 != 0:
            logger.error(
                f"File {filename} has an odd number of bytes, cannot read as I/Q pairs."
            )
            return None

        # Unpack all float values
        float_format = f"<{num_floats}f"
        float_values = struct.unpack(
            float_format, file_data[: num_floats * 4]
        )  # Each float is 4 bytes

        # Convert to complex I/Q pairs
        i_samples = np.array(float_values[0::2])
        q_samples = np.array(float_values[1::2])

        iq_data = i_samples + 1j * q_samples

        logger.debug(f"Read {len(iq_data)} I/Q samples from {filename}")

        return iq_data
