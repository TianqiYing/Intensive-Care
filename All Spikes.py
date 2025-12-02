import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

delay = 300  # default start index for most files

def process_file(i):
    # -------------------------
    # decide offset per file
    # -------------------------
    if i == 7:
        offset = 2000   # your special case
    else:
        offset = delay

    filename = f"N{i}.xlsx"
    print(f"\n=== Processing {filename} (offset={offset}) ===")

    # Load data
    df = pd.read_excel(filename)

    # Convert time column
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")

    # Extract temperature & apply offset
    temp = df["Temperature"].astype(float).values
    data = temp[offset:]

    # Spike detection
    median = np.nanmedian(data)
    mad = np.nanmedian(np.abs(data - median))
    modified_z = 0.6745 * (data - median) / (mad + 1e-9)

    spike_indices = np.where(modified_z < -3)[0]

    if len(spike_indices) == 0:
        print("No spikes detected.")
        return np.nan, pd.NaT, pd.NaT  # so averages & plotting don't break

    # Compute spike times (map back to original df index)
    start_idx = spike_indices[0] + offset + 1
    end_idx   = spike_indices[-1] + offset + 1

    time1 = df.loc[start_idx, "Time"]
    time2 = df.loc[end_idx, "Time"]
    total = time2 - time1  # Timedelta

    print(f"Spikes detected: {len(spike_indices)}")
    print(f"Start time: {time1}")
    print(f"End time:   {time2}")
    print(f"Total spike duration: {total}")

    return total, time1, time2


# ============================
#   PROCESS ALL 16 FILES
# ============================

results = []      # Timedeltas (or NaN)
intervals = []    # (start_time, end_time) for plotting

for i in range(1, 17):
    total, t_start, t_end = process_file(i)
    results.append(total)
    intervals.append((t_start, t_end))

print("\n=== All results (Timedeltas) ===")
for i, t in enumerate(results, start=1):
    print(f"N{i}: {t}")


# ============================
#   AVERAGING FIRST 8 / LAST 8
# ============================

# Convert Timedeltas → seconds for averaging
secs = np.array([
    t.total_seconds() if pd.notna(t) else np.nan
    for t in results
])

avg_first8_sec = np.nanmean(secs[:8])
avg_last8_sec  = np.nanmean(secs[8:])

# Convert back to Timedelta for reporting
avg_first8 = pd.to_timedelta(avg_first8_sec, unit="s")
avg_last8  = pd.to_timedelta(avg_last8_sec, unit="s")

print("\n=== Averages ===")
print("Average first 8 spike times :", avg_first8)
print("Average last 8 spike times  :", avg_last8)
print("No difference between male and female torpor")


# ============================
#   TIMELINE PLOT FOR ALL 16
# ============================

fig, ax = plt.subplots(figsize=(12, 6))

for i, (start, end) in enumerate(intervals, start=1):
    if pd.isna(start) or pd.isna(end):
        continue  # skip files with no spikes
    # draw a horizontal bar/line for each file
    ax.plot([start, end], [i, i], linewidth=6)

ax.set_yticks(range(1, 17))
ax.set_yticklabels([f"N{i}" for i in range(1, 17)])
ax.set_ylabel("File")
ax.set_xlabel("Time")
ax.set_title("Spike intervals for N1–N16")

# ===== x-axis every hour =====
ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))  # tick every hour
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))

fig.autofmt_xdate()  # rotate date labels nicely
plt.tight_layout()
plt.savefig("Mice torpor timeline.pdf")
plt.show()
