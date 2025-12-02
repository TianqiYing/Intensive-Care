import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
delay = 300
df = pd.read_excel("N1.xlsx")
x = df["Temperature"].astype(float).values
x = x[delay:]

# Convert to array
data = np.array(x)


median = np.nanmedian(data)
mad = np.nanmedian(np.abs(data - median))

print(median, mad)
# Modified Z-score (no manual threshold needed)
# 3.5 is mathematically suggested by literature (Iglewicz & Hoaglin)
modified_z = 0.6745 * (data - median) / (mad + 1e-9)

# Automatically detect spikes
spike_indices = np.where(modified_z < -3)[0]

print("Detected spikes:", spike_indices+delay+1)
print("Spike values:", data[spike_indices])

# Optional: plot



df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
if len(spike_indices) > 0:
    time1 = df.loc[spike_indices[0]+delay+1, "Time"]
    time2 = df.loc[spike_indices[-1]+delay+1, "Time"]
    print(time1, time2)
    total = time2 - time1
    print(total)

    plt.figure(figsize=(12,5))
    plt.plot(data, label="Temperature")
    plt.axvspan(
        spike_indices[0],
        spike_indices[-1],
        color='red',
        alpha=0.3,
        label=f"Total spike time is {total}"
    )
    plt.xlim(11000, 12000)
    plt.legend()
    plt.savefig("Mice torpor N1 Zoomed.pdf")
    plt.show()
else:
    plt.figure(figsize=(12, 5))
    plt.plot(data, label="Temperature")
    plt.legend()
    plt.show()
