"""
analysis.py
"""

from collection import OUTPUT_DIRECTORY
from rx_data_analyser import RxConfig, RxDataAnalyser
from sweep_data_analyser import SweepConfig, SweepDataAnalyser

# from data.collection.dual_sdr_collection import OUTPUT_DIRECTORY

if __name__ == "__main__":
    rx_config = RxConfig(
        sample_rate_hz=20000000,  # Sample rate in Hz
        filename=f"{OUTPUT_DIRECTORY}/rx/0.iq",
    )

    rx_data_analyser = RxDataAnalyser(rx_config)
    rx_data_analyser.compute()

    sweep_config = SweepConfig(
        filename=f"{OUTPUT_DIRECTORY}/sweep/0.bin",
    )

    sweep_data_analyser = SweepDataAnalyser(sweep_config)
    sweep_data_analyser.compute()
