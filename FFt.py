import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Load temperature data
# ---------------------------------------------------------
df = pd.read_excel("N1.xlsx")  # change path if needed

temp = df["Temperature"].values

# ---------------------------------------------------------
# 2. Extract the section of interest: samples 2000–3000
# ---------------------------------------------------------
section = temp
N = len(section)

# Sampling interval (1 minute = 60 seconds)
dt = 60
fs = 1 / dt   # sampling frequency (Hz)

# ---------------------------------------------------------
# 3. Detrend and apply window (Hann)
# ---------------------------------------------------------
section_d = section - np.mean(section)     # remove DC offset
window = np.hanning(N)
xw = section_d * window

# ---------------------------------------------------------
# 4. Compute FFT
# ---------------------------------------------------------
fft_vals = np.fft.fft(xw)
freqs = np.fft.fftfreq(N, d=dt)

# Keep positive frequencies (single-sided spectrum)
half = N // 2
freqs_half = freqs[:half]
amp_half = (2 / N) * np.abs(fft_vals[:half])

# Convert frequency axis:
# cycles per hour = Hz * 3600
freq_hr = freqs_half * 3600

# ---------------------------------------------------------
# 5. Plot the FFT spectrum
# ---------------------------------------------------------
plt.figure(figsize=(12, 6))
plt.plot(freq_hr, amp_half, color="orange")
plt.xlim(0, 5)
plt.xlabel("Frequency (cycles per hour)")
plt.ylabel("Amplitude")
plt.title("FFT Amplitude Spectrum")
plt.grid(True)

plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------
df = pd.read_excel("N1.xlsx")       # change path if needed
temp = df["Temperature"].values
N = len(temp)

# ---------------------------------------------------------
# 2. Sampling parameters
# ---------------------------------------------------------
dt = 60          # 1 minute = 60 seconds
fs = 1/dt        # sampling frequency (Hz)

# ---------------------------------------------------------
# 3. Detrend + window
# ---------------------------------------------------------
temp_d = temp - np.mean(temp)       # remove DC offset
window = np.hanning(N)
xw = temp_d * window

# ---------------------------------------------------------
# 4. FFT
# ---------------------------------------------------------
fft_vals = np.fft.fft(xw)
freqs = np.fft.fftfreq(N, d=dt)

# Keep only positive frequencies
half = N // 2
freqs_half = freqs[:half]
amp_half = (2/N) * np.abs(fft_vals[:half])

# Convert to cycles per hour
freq_hr = freqs_half * 3600

# ---------------------------------------------------------
# 5. Convert frequency -> period (hours)
# ---------------------------------------------------------
valid = freq_hr > 0               # avoid divide-by-zero
period_hours = 1 / freq_hr[valid]
amp_valid = amp_half[valid]

# ---------------------------------------------------------
# 6. Plot Amplitude vs Period
# ---------------------------------------------------------
plt.figure(figsize=(12,6))
plt.plot(period_hours, amp_valid, color='orange')
plt.xlabel("Period (hours)")
plt.ylabel("Amplitude")
plt.title("FFT Amplitude vs Period (Entire Dataset)")
plt.grid(True)
plt.xlim(0, 100)   # limit to 100 hours for readability
plt.show()
