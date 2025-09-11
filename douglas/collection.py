import os
import subprocess
import time
import traceback
from dataclasses import dataclass
from typing import Literal

from app.constants import (
    AMP_ENABLE,
    BIN_WIDTH_HZ,
    FREQUENCY_HZ,
    HACKRF_BUFFER_CLEAR_CMD,
    LNA_GAIN,
    NUM_SAMPLES,
    NUM_SWEEPS,
    SAMPLE_RATE_HZ,
    START_FREQ_MHZ,
    STOP_FREQ_MHZ,
    VGA_GAIN,
)
from app.modules.logging.logger import logger

# Folder structure and naming convention for output files:
# data/output/
# └─ [ExperimentName]_[Location]_[Date]/
#    └─ [DroneType]_[Mode]_[Distance]_[GainSettings]_[Index]/
#       ├─ rx/
#       │  └─ [RunNumber].iq
#       └─ sweep/
#          └─ [RunNumber].bin
DATA_DIR = "data/output/DatasetCollection_HCJCField_110925"
MODE = "Ifft"  # Mode depends on the specific experiment
DISTANCE = "0m"  # Distance depends on the specific experiment
INDEX = "01"
OUTPUT_DIRECTORY = (
    f"{DATA_DIR}/Background_{MODE}_{DISTANCE}_"
    f"{AMP_ENABLE}_{LNA_GAIN}_{VGA_GAIN}_{INDEX}"
)

NUM_RUNS = 5  # Number of runs to perform


@dataclass
class Config:
    start_freq_mhz: int  # Start freq for hackrf_sweep
    stop_freq_mhz: int  # Stop freq for hackrf_sweep
    frequency_hz: int  # frequency for hackrf_transfer
    amp_enable: int
    lna_gain: int
    vga_gain: int
    num_sweeps: int
    bin_width_hz: int
    sample_rate: int
    num_samples: int
    output_directory: str
    sweep_capture_mode: Literal["binary", "ifft"] = "ifft"


class DataCollector:
    def __init__(self, config: Config):
        self.config = config

        self._validate_parameters()

        if not os.path.exists(self.config.output_directory):
            os.makedirs(self.config.output_directory)

        # Create subdirectories for sweep and RX data
        os.makedirs(os.path.join(self.config.output_directory, "sweep"), exist_ok=True)
        os.makedirs(os.path.join(self.config.output_directory, "rx"), exist_ok=True)

    def _validate_parameters(self):
        pass  # TODO

    def _build_sweep_command(self, filename: str):
        start_freq_mhz = self.config.start_freq_mhz
        stop_freq_mhz = self.config.stop_freq_mhz

        return [
            "hackrf_sweep",
            "-f",
            f"{start_freq_mhz}:{stop_freq_mhz}",
            "-B" if self.config.sweep_capture_mode == "binary" else "-I",
            "-a",
            str(self.config.amp_enable),
            "-l",
            str(self.config.lna_gain),
            "-g",
            str(self.config.vga_gain),
            "-w",
            str(self.config.bin_width_hz),
            "-N",
            str(self.config.num_sweeps),
            "-r",
            filename,
        ]

    def _build_rx_command(self, filename: str):
        return [
            "hackrf_transfer",
            "-f",
            str(self.config.frequency_hz),
            "-a",
            str(self.config.amp_enable),
            "-l",
            str(self.config.lna_gain),
            "-g",
            str(self.config.vga_gain),
            "-s",
            str(self.config.sample_rate),
            "-n",
            str(self.config.num_samples),
            "-r",
            str(filename),
        ]

    def sweep(self, filename: str = ""):
        cmd = self._build_sweep_command(filename)
        logger.info(f"Running command: {' '.join(cmd)}")

        self.process = subprocess.run(cmd)

    def rx(self, filename: str = ""):
        cmd = self._build_rx_command(filename)
        logger.info(f"Running command: {' '.join(cmd)}")
        self.process = subprocess.run(cmd)

    def run(self, num_runs: int = 1):
        for i in range(num_runs):
            sfx = "bin" if self.config.sweep_capture_mode == "binary" else "ifft"
            sweep_filename = f"{self.config.output_directory}/sweep/{i}.{sfx}"
            rx_filename = f"{self.config.output_directory}/rx/{i}.iq"

            logger.info(f"Running sweep {i + 1}/{num_runs}")
            self.sweep(sweep_filename)

            self.cleanup()

            time.sleep(1)

            logger.info(f"Running RX {i + 1}/{num_runs}")
            self.rx(rx_filename)

    def cleanup(self):
        try:
            logger.info("Clearing HackRF buffer...")
            subprocess.run(
                HACKRF_BUFFER_CLEAR_CMD,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                timeout=1,
            )
        except (subprocess.TimeoutExpired, Exception) as e:
            logger.warning(
                f"Failed to clear HackRF buffer: {e}"
                f"Traceback: {traceback.format_exc()}"
            )


if __name__ == "__main__":
    config = Config(
        start_freq_mhz=START_FREQ_MHZ,
        stop_freq_mhz=STOP_FREQ_MHZ,
        frequency_hz=FREQUENCY_HZ,
        amp_enable=AMP_ENABLE,
        lna_gain=LNA_GAIN,
        vga_gain=VGA_GAIN,
        num_sweeps=NUM_SWEEPS,
        bin_width_hz=BIN_WIDTH_HZ,
        sample_rate=SAMPLE_RATE_HZ,
        num_samples=NUM_SAMPLES,
        output_directory=OUTPUT_DIRECTORY,
    )

    data_collector = DataCollector(config)
    data_collector.run(num_runs=NUM_RUNS)
