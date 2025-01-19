import torch
import matplotlib.pyplot as plt
import numpy as np

# Function to compute the Fourier transform
def fast_fourier_transform(signal, sampling_rate):
    n = signal.size(0)
    freq = torch.fft.fftfreq(n, 1/sampling_rate)  # Frequency bins
    fft = torch.fft.fft(signal)  # Compute FFT
    return freq, fft

sampling_rate = 1000  
duration = 1.0  
t = torch.arange(0, duration, 1/sampling_rate, dtype=torch.float32)
signal = torch.sin(2 * torch.pi * 50 * t) + 0.5 * torch.sin(2 * torch.pi * 120 * t)  # 50 Hz + 120 Hz

# Add some noise to the signal
signal += 0.2 * torch.randn_like(signal)

# Compute the Fourier transform
freq, fft = fast_fourier_transform(signal, sampling_rate)

# Plot the time-domain signal
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title("Time-Domain Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)

# Plot the frequency-domain spectrum
plt.subplot(2, 1, 2)
plt.plot(freq[:len(freq)//2], torch.abs(fft[:len(fft)//2]))  # Plot only positive frequencies
plt.title("Frequency-Domain Spectrum")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.grid(True)

plt.tight_layout()
plt.show()