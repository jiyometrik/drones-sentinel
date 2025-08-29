import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# i was bored so i chatgpted this during physics class

# Folder with .ifft files
folder_path = r"C:\Users\aksha\OneDrive\VSC\VSC\Work\DroneResearch\data\output\DatasetCollection_Scis2Outdoors_210825\Aquila16_Ifft_50m_1_40_40_01\sweep"
file_paths = glob.glob(os.path.join(folder_path, "*.ifft"))

def load_ifft(path):
    """Try different dtypes to load .ifft files."""
    dtypes_to_try = [np.float32, np.float64, np.complex64, np.complex128]
    for dt in dtypes_to_try:
        try:
            data = np.fromfile(path, dtype=dt)
            if data.size > 0:
                print(f"[OK] {os.path.basename(path)} loaded with dtype={dt}, shape={data.shape}")
                return data, dt
        except Exception as e:
            print(f"Failed {dt} for {path}: {e}")
    raise ValueError(f"Could not interpret {path} with common dtypes.")

for path in file_paths:
    data, dtype_used = load_ifft(path)

    # --- 1. Time-domain signal ---
    plt.figure(figsize=(12,4))
    if np.iscomplexobj(data):
        plt.plot(np.real(data), alpha=0.7, label="Real")
        plt.plot(np.imag(data), alpha=0.7, label="Imag")
    else:
        plt.plot(data, label="Signal")
    plt.legend()
    plt.title(f"Time-Domain Signal: {os.path.basename(path)}")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.show()

    # --- 2. Histogram ---
    plt.figure(figsize=(6,4))
    plt.hist(np.real(data), bins=100, alpha=0.7, label="Real")
    if np.iscomplexobj(data):
        plt.hist(np.imag(data), bins=100, alpha=0.7, label="Imag")
    plt.title("Amplitude Distribution")
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.legend()
    plt.show()

    # --- 3. Spectrogram ---
    plt.figure(figsize=(10,4))
    plt.specgram(np.real(data), Fs=1000, NFFT=256, noverlap=128)  # adjust Fs if known
    plt.title("Spectrogram")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="Power (dB)")
    plt.show()

    # --- 4. Frequency Spectrum ---
    fft_vals = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(data))
    plt.figure(figsize=(10,4))
    plt.plot(freqs, np.abs(fft_vals))
    plt.title("Frequency Spectrum")
    plt.xlabel("Frequency (normalized)")
    plt.ylabel("Magnitude")
    plt.show()

    # --- 5. IQ Scatter Plot ---
    if np.iscomplexobj(data):
        plt.figure(figsize=(5,5))
        plt.scatter(np.real(data), np.imag(data), s=2, alpha=0.5)
        plt.title("IQ Scatter Plot")
        plt.xlabel("In-phase (I)")
        plt.ylabel("Quadrature (Q)")
        plt.axis("equal")
        plt.show()

    # --- 6. Cumulative Energy ---
    energy = np.cumsum(np.abs(data)**2)
    plt.figure(figsize=(10,4))
    plt.plot(energy)
    plt.title("Cumulative Energy")
    plt.xlabel("Sample")
    plt.ylabel("Energy")
    plt.show()

    # --- 7. Autocorrelation ---
    corr = np.correlate(data, data, mode='full')
    corr = corr[len(corr)//2:]  # keep positive lags
    plt.figure(figsize=(10,4))
    plt.plot(corr)
    plt.title("Autocorrelation")
    plt.xlabel("Lag")
    plt.ylabel("Correlation")
    plt.show()
