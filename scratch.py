"""
scratch.py
Just a file to test out some methods
"""

import os

import src.constants as cts
import src.preprocessing as prep

testfile = os.path.join(
    cts.DATADIR,
    "DataCollection_HCJCTrack_110925/Background_Ifft_5750_0m_1_40_40_01/sweep/10.ifft",
)
s = prep.read_raw_ifft(testfile)
print(s)
f, t, Zxx = prep.compute_stft(s, representation="magnitude")
print(f"{f = }", f"{t = }", f"{Zxx = }", sep="\n")
print(f"\n{f.shape = }, {t.shape = }, {Zxx.shape = }")

# pad spectrogram
Zxx_padded = prep.pad_spectrogram(
    Zxx,
    target_freq=cts.TARGET_FREQ,
    target_time=cts.TARGET_TIME,
)
print(f"\n{Zxx = }\n{Zxx_padded.shape = }")
