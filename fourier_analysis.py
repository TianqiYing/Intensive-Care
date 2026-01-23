"""
FINAL ANALYSIS - FOURIER + DELAY EMBEDDINGS
============================================

Clean, verified approach:
- 48h windows for Fourier (proper 24h detection)
- Delay embeddings with tau=6h
- Removes corrupted data
- Handles different sampling rates
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

print("=" * 70)
print("FOURIER + DELAY EMBEDDING ANALYSIS")
print("=" * 70)

# ==========================================
# SETTINGS
# ==========================================

DATA_FOLDER = r"C:\Users\harim\PycharmProjects\SVM"  # CHANGE TO YOUR PATH

# Fourier settings
WINDOW_HOURS = 48  # Need >= 48h to properly detect 24h rhythms
STEP_HOURS = 12  # Move every 12h

# Delay embedding settings
TAU_HOURS = 6  # From 24h/4 = 6h
EMBEDDING_DIM = 3
DOWNSAMPLE_MINUTES = 15  # Keep file size reasonable

CALIB_HOURS = 3

print(f"\nSettings:")
print(f"  Fourier window: {WINDOW_HOURS}h, step: {STEP_HOURS}h")
print(f"  Delay embedding: tau={TAU_HOURS}h, dim={EMBEDDING_DIM}")
print(f"  Downsample embeddings: every {DOWNSAMPLE_MINUTES}min")


# ==========================================
# FUNCTIONS
# ==========================================

def clean_data(df, mouse_id):
    """Clean data - remove bad timestamps"""
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    df = df.dropna(subset=['Time'])

    # Remove wrong years (2038 bug)
    df = df[df['Time'].dt.year == 2023].copy()

    df = df.sort_values('Time').reset_index(drop=True)
    df = df.drop_duplicates(subset=['Time'], keep='first')

    return df


def process_one_mouse(filepath, mouse_id):
    """Process one mouse - extract both Fourier and embeddings"""

    print(f"\n{mouse_id:4s}: ", end="")

    try:
        # Load
        df = pd.read_excel(filepath)

        if 'Time' not in df.columns or 'Temperature' not in df.columns:
            print("Missing columns")
            return None, None

        # Clean
        df = clean_data(df, mouse_id)

        if len(df) < 100:
            print(f"Only {len(df)} points")
            return None, None

        # Check duration
        duration_days = (df['Time'].max() - df['Time'].min()).days

        if duration_days < 7 or duration_days > 100:
            print(f"Bad duration: {duration_days} days")
            return None, None

        # Remove calibration
        cutoff = df['Time'].min() + pd.Timedelta(hours=CALIB_HOURS)
        df = df[df['Time'] >= cutoff].copy()

        # Determine sampling rate
        time_diffs = df['Time'].diff().dt.total_seconds().dropna()
        median_interval_sec = time_diffs.median()

        if median_interval_sec < 90:
            resample_rule = '1min'
            d_minutes = 1.0
        else:
            resample_rule = '5min'
            d_minutes = 5.0

        # Resample to regular grid
        df = df.set_index('Time').resample(resample_rule).mean()

        # Interpolate missing values (max 1 hour gap)
        max_gap = 12 if resample_rule == '5min' else 60
        df['Temperature'] = df['Temperature'].interpolate(limit=max_gap)
        df = df.dropna()

        if len(df) < (WINDOW_HOURS * 60 / d_minutes):
            print(f"Insufficient data")
            return None, None

        df = df.reset_index()

        temperature = df['Temperature'].values
        times = df['Time'].values

        # ==========================================
        # PART 1: FOURIER FEATURES
        # ==========================================

        window_size = int((WINDOW_HOURS * 60) / d_minutes)
        step_size = int((STEP_HOURS * 60) / d_minutes)

        fourier_features = []

        for start_idx in range(0, len(temperature) - window_size, step_size):
            end_idx = start_idx + window_size

            window_temp = temperature[start_idx:end_idx]
            window_time = times[start_idx + window_size // 2]

            # Temperature stats
            mean_temp = float(np.mean(window_temp))
            std_temp = float(np.std(window_temp))
            min_temp = float(np.min(window_temp))
            max_temp = float(np.max(window_temp))

            # FFT
            x = window_temp - np.mean(window_temp)

            if np.std(x) < 0.01:
                # Flat signal
                frac_circ = 0.0
                frac_ultra = 0.0
                dom_period = np.nan
            else:
                fft_vals = fft(x)
                freqs = fftfreq(len(x), d=d_minutes)

                pos_mask = freqs > 0
                freqs_pos = freqs[pos_mask]
                power = np.abs(fft_vals[pos_mask]) ** 2

                periods_hours = (1 / freqs_pos) / 60

                # Band power
                def band_power(periods, pwr, low, high):
                    mask = (periods >= low) & (periods <= high)
                    return float(np.sum(pwr[mask]))

                P_circ = band_power(periods_hours, power, 20, 28)
                P_ultra = band_power(periods_hours, power, 4, 8)
                P_total = band_power(periods_hours, power, 2, 48)

                if P_total > 0:
                    frac_circ = P_circ / P_total
                    frac_ultra = P_ultra / P_total
                else:
                    frac_circ = 0.0
                    frac_ultra = 0.0

                # Dominant period
                relevant_mask = (periods_hours >= 2) & (periods_hours <= 48)
                if np.sum(relevant_mask) > 0:
                    dom_idx = np.argmax(power[relevant_mask])
                    dom_period = float(periods_hours[relevant_mask][dom_idx])
                else:
                    dom_period = np.nan

            fourier_features.append({
                'mouse_id': mouse_id,
                'window_time': window_time,
                'mean_temp': mean_temp,
                'std_temp': std_temp,
                'min_temp': min_temp,
                'max_temp': max_temp,
                'circadian_strength': frac_circ,
                'ultradian_strength': frac_ultra,
                'dominant_period_h': dom_period,
            })

        fourier_df = pd.DataFrame(fourier_features)

        # ==========================================
        # PART 2: DELAY EMBEDDINGS
        # ==========================================

        # Z-score temperature
        T_mean = temperature.mean()
        T_std = temperature.std()
        T_z = (temperature - T_mean) / (T_std + 1e-10)

        # Delay parameters in samples
        tau = int((TAU_HOURS * 60) / d_minutes)
        m = EMBEDDING_DIM
        max_lag = (m - 1) * tau

        if len(T_z) > max_lag:
            indices = np.arange(max_lag, len(T_z))

            # Create embedding
            X_cols = []
            for k in range(m):
                lag = k * tau
                X_cols.append(T_z[indices - lag])

            X = np.vstack(X_cols).T
            t_embed = times[indices]

            # Downsample
            downsample_step = int(DOWNSAMPLE_MINUTES / d_minutes)
            X = X[::downsample_step]
            t_embed = t_embed[::downsample_step]

            embeddings_df = pd.DataFrame({
                'mouse_id': mouse_id,
                'time': t_embed,
                'X0': X[:, 0],
                'X1': X[:, 1],
                'X2': X[:, 2],
            })
        else:
            embeddings_df = pd.DataFrame()

        print(f"{len(fourier_df)} windows, {len(embeddings_df)} embed points ({duration_days} days)")

        return fourier_df, embeddings_df

    except Exception as e:
        print(f"ERROR: {e}")
        return None, None


# ==========================================
# PROCESS ALL MICE
# ==========================================

print("\n" + "=" * 70)
print("PROCESSING")
print("=" * 70)

data_folder = Path(DATA_FOLDER)
excel_files = sorted(list(data_folder.glob("N*.xlsx")))

print(f"\nFound {len(excel_files)} files")

all_fourier = []
all_embeddings = []

for filepath in excel_files:
    mouse_id = filepath.stem

    fourier_df, embeddings_df = process_one_mouse(filepath, mouse_id)

    if fourier_df is not None:
        all_fourier.append(fourier_df)

    if embeddings_df is not None and len(embeddings_df) > 0:
        all_embeddings.append(embeddings_df)

if len(all_fourier) == 0:
    print("\nERROR: No data processed!")
    exit(1)

# Combine
fourier_all = pd.concat(all_fourier, ignore_index=True)
embeddings_all = pd.concat(all_embeddings, ignore_index=True) if len(all_embeddings) > 0 else pd.DataFrame()

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\nProcessed: {len(all_fourier)} mice")
print(f"Fourier windows: {len(fourier_all)}")
print(f"Embedding points: {len(embeddings_all)}")

# Per-mouse summary
per_mouse = fourier_all.groupby('mouse_id').agg({
    'circadian_strength': ['mean', 'std', 'min', 'max'],
    'ultradian_strength': 'mean',
    'mean_temp': 'mean',
    'window_time': 'count'
}).round(3)

per_mouse.columns = ['circ_mean', 'circ_std', 'circ_min', 'circ_max', 'ultra_mean', 'temp', 'n_windows']
per_mouse = per_mouse.reset_index().sort_values('mouse_id')

print("\n" + "-" * 70)
print("Per-mouse summary:")
print("-" * 70)
print(per_mouse.to_string(index=False))

# Overall
print("\n" + "-" * 70)
print("Overall statistics:")
print("-" * 70)
print(f"Circadian: {fourier_all['circadian_strength'].mean():.3f} ± {fourier_all['circadian_strength'].std():.3f}")
print(f"Range: {fourier_all['circadian_strength'].min():.3f} to {fourier_all['circadian_strength'].max():.3f}")

# Distribution
high = (fourier_all['circadian_strength'] > 0.6).sum()
med = ((fourier_all['circadian_strength'] >= 0.3) & (fourier_all['circadian_strength'] <= 0.6)).sum()
low = (fourier_all['circadian_strength'] < 0.3).sum()
total = len(fourier_all)

print(f"\nWindows by circadian strength:")
print(f"  High (>0.6):    {high:4d} ({100 * high / total:5.1f}%)")
print(f"  Medium (0.3-0.6): {med:4d} ({100 * med / total:5.1f}%)")
print(f"  Low (<0.3):     {low:4d} ({100 * low / total:5.1f}%)")

# ==========================================
# SAVE FILES
# ==========================================

print("\n" + "=" * 70)
print("SAVING")
print("=" * 70)

# Save Fourier features
fourier_all.to_csv('fourier_features_CLEAN.csv', index=False)
print("\nSaved: fourier_features_CLEAN.csv")
print(f"  {len(fourier_all)} rows, {len(fourier_all.columns)} columns")

# Save embeddings
if len(embeddings_all) > 0:
    embeddings_all.to_csv('delay_embeddings_CLEAN.csv', index=False)
    print("\nSaved: delay_embeddings_CLEAN.csv")
    print(f"  {len(embeddings_all)} rows, {len(embeddings_all.columns)} columns")

# Save per-mouse summary
per_mouse.to_csv('per_mouse_summary.csv', index=False)
print("\nSaved: per_mouse_summary.csv")

# ==========================================
# VISUALIZATIONS
# ==========================================

print("\n" + "=" * 70)
print("CREATING VISUALIZATIONS")
print("=" * 70)

mice = sorted(fourier_all['mouse_id'].unique())
n_mice = len(mice)

# Plot 1: Circadian timeseries
n_cols = 4
n_rows = (n_mice + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
axes = axes.flatten() if n_mice > 1 else [axes]

for i, mouse_id in enumerate(mice):
    ax = axes[i]
    mouse_data = fourier_all[fourier_all['mouse_id'] == mouse_id].sort_values('window_time')

    ax.plot(mouse_data['window_time'], mouse_data['circadian_strength'],
            'o-', linewidth=1.5, markersize=4, alpha=0.7)

    ax.axhline(0.6, color='green', linestyle='--', alpha=0.3, linewidth=1)
    ax.axhline(0.3, color='red', linestyle='--', alpha=0.3, linewidth=1)

    avg = mouse_data['circadian_strength'].mean()
    ax.set_ylabel('Circadian strength')
    ax.set_title(f'{mouse_id} (avg={avg:.2f})')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)

for i in range(len(mice), len(axes)):
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('circadian_timeseries.png', dpi=200)
print("Saved: circadian_timeseries.png")
plt.close()

# Plot 2: Circadian vs Temperature
fig, ax = plt.subplots(figsize=(10, 6))

for mouse_id in mice:
    mouse_data = fourier_all[fourier_all['mouse_id'] == mouse_id]
    ax.scatter(mouse_data['circadian_strength'], mouse_data['mean_temp'],
               alpha=0.5, s=30, label=mouse_id)

ax.set_xlabel('Circadian Strength')
ax.set_ylabel('Mean Temperature (°C)')
ax.set_title('Circadian Strength vs Temperature')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('circadian_vs_temperature.png', dpi=200)
print("Saved: circadian_vs_temperature.png")
plt.close()

# Plot 3: Example embedding (first mouse)
if len(embeddings_all) > 0:
    example_mouse = mice[0]
    example_embed = embeddings_all[embeddings_all['mouse_id'] == example_mouse]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    c = np.linspace(0, 1, len(example_embed))
    scatter = ax.scatter(example_embed['X0'], example_embed['X1'], example_embed['X2'],
                         c=c, cmap='viridis', s=2, alpha=0.6)

    ax.set_xlabel('T(t)')
    ax.set_ylabel(f'T(t-{TAU_HOURS}h)')
    ax.set_zlabel(f'T(t-{2 * TAU_HOURS}h)')
    ax.set_title(f'Delay Embedding: {example_mouse}')
    plt.colorbar(scatter, label='Time progression', shrink=0.5)

    plt.tight_layout()
    plt.savefig('example_delay_embedding.png', dpi=200)
    print("Saved: example_delay_embedding.png")
    plt.close()

print("\n" + "=" * 70)
print("COMPLETE!")
print("=" * 70)

print(f"""
Files created:
1. fourier_features_CLEAN.csv - {len(fourier_all)} windows for clustering
2. delay_embeddings_CLEAN.csv - {len(embeddings_all)} points for clustering
3. per_mouse_summary.csv - Quick overview
4. circadian_timeseries.png - All mice over time
5. circadian_vs_temperature.png - Relationship plot
6. example_delay_embedding.png - 3D embedding visualization

Ready for analysis!
""")

# ==========================================
# PRESENTATION / EXPLANATION PLOTS (ADD-ON)
# ==========================================

print("\n" + "=" * 70)
print("PRESENTATION PLOTS + METRIC KEY")
print("=" * 70)

import matplotlib.dates as mdates
from sklearn.decomposition import PCA

# --------------------------
# 0) METRIC KEY (print + save)
# --------------------------
metric_key = f"""
METRIC KEY (how to interpret the numbers)
-----------------------------------------

circadian_strength:
  - Definition (in this script): fraction of FFT power (2–48h) that lies in the 20–28h band.
  - Range: 0 to 1.
  - Interpretation:
      0.0  = essentially no 24h structure in that window
      0.3  = weak/moderate circadian structure
      0.6+ = strong circadian structure
  - Note: This is a *window-level* measure; it can fluctuate day-to-day even in normal animals.

ultradian_strength:
  - Definition: fraction of FFT power (2–48h) that lies in the 4–8h band.
  - Interpretation: higher values suggest stronger multi-hour structure (bouts/harmonics/fragmentation).

dominant_period_h:
  - Definition: the single period (2–48h) with maximal power within a window.
  - Interpretation: often ~24h in stable circadian animals; drift/spread can occur with nonstationarity.

circ_mean / circ_std / circ_min / circ_max (per_mouse_summary.csv):
  - These summarize circadian_strength across all windows for a mouse.
  - Example: circ_mean=0.60 means: “on average, 60% of that mouse’s 2–48h power sits in 20–28h”.
  - circ_std tells you how stable that measure is across time.

temp (per_mouse summary):
  - mean of mean_temp across windows (a baseline operating level, not a state label).

Why this helps clustering:
  - Instead of clustering raw temperature points, you can cluster *windows* or *embedded points*:
      - windows -> rhythm fingerprints (circadian/ultradian structure)
      - embeddings -> reconstructed state geometry (attractor regions + transitions)
"""

print(metric_key)

# Save the key as a simple image (useful for slides)
fig = plt.figure(figsize=(12, 6))
fig.text(0.02, 0.98, metric_key, va='top', family='monospace', fontsize=10)
plt.axis('off')
plt.tight_layout()
plt.savefig("metric_key.png", dpi=200)
plt.close()
print("Saved: metric_key.png")

# Ensure time column is datetime
fourier_all["window_time"] = pd.to_datetime(fourier_all["window_time"])

# --------------------------
# 1) HEATMAP: circadian strength over time (all mice)
# --------------------------
# This is usually the single best “group meeting” plot.
heat = fourier_all.pivot_table(
    index="mouse_id",
    columns="window_time",
    values="circadian_strength",
    aggfunc="mean"
).sort_index()

fig, ax = plt.subplots(figsize=(16, 0.6 * len(heat) + 2))
im = ax.imshow(heat.values, aspect='auto', interpolation='nearest')

ax.set_yticks(np.arange(len(heat.index)))
ax.set_yticklabels(heat.index)
ax.set_title("Circadian strength heatmap (per 48h window, step=12h)")
ax.set_xlabel("Time")

# Show fewer x labels nicely
x_ticks = np.linspace(0, heat.shape[1]-1, 8).astype(int)
ax.set_xticks(x_ticks)
ax.set_xticklabels([heat.columns[i].strftime("%m-%d") for i in x_ticks], rotation=45, ha='right')

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("circadian_strength (fraction of 2–48h power in 20–28h band)")

plt.tight_layout()
plt.savefig("heatmap_circadian_strength.png", dpi=200)
plt.close()
print("Saved: heatmap_circadian_strength.png")

# --------------------------
# 2) DISTRIBUTION: what does circ_mean=0.6 mean?
# --------------------------
# Show distribution of window-level circadian_strength across all mice
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(fourier_all["circadian_strength"], bins=30, alpha=0.8)
ax.axvline(0.3, linestyle="--", linewidth=2, alpha=0.7)
ax.axvline(0.6, linestyle="--", linewidth=2, alpha=0.7)
ax.set_title("Distribution of window-level circadian_strength (all mice, all windows)")
ax.set_xlabel("circadian_strength")
ax.set_ylabel("count of windows")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("dist_circadian_strength_windows.png", dpi=200)
plt.close()
print("Saved: dist_circadian_strength_windows.png")

# Also show per-mouse circ_mean distribution
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(per_mouse["mouse_id"], per_mouse["circ_mean"])
ax.axhline(0.3, linestyle="--", linewidth=2, alpha=0.7)
ax.axhline(0.6, linestyle="--", linewidth=2, alpha=0.7)
ax.set_title("Per-mouse circ_mean (average circadian strength)")
ax.set_xlabel("mouse_id")
ax.set_ylabel("circ_mean")
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("per_mouse_circ_mean_bar.png", dpi=200)
plt.close()
print("Saved: per_mouse_circ_mean_bar.png")

# --------------------------
# 3) STRUCTURE PLOT: circadian vs ultradian (windows)
# --------------------------
# Shows whether mice/windows form “regimes” in rhythm space
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(
    fourier_all["circadian_strength"],
    fourier_all["ultradian_strength"],
    s=20,
    alpha=0.4
)
ax.set_title("Windows in rhythm-space: circadian vs ultradian strength")
ax.set_xlabel("circadian_strength (20–28h fraction)")
ax.set_ylabel("ultradian_strength (4–8h fraction)")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("scatter_circ_vs_ultra.png", dpi=200)
plt.close()
print("Saved: scatter_circ_vs_ultra.png")

# --------------------------
# 4) EMBEDDING VISUAL: 2D PCA of delay embedding for one mouse (more interpretable than 3D)
# --------------------------
if len(embeddings_all) > 0:
    example_mouse = mice[0]  # keep your existing choice, or set e.g. "N6"
    ex = embeddings_all[embeddings_all["mouse_id"] == example_mouse].copy()
    ex["time"] = pd.to_datetime(ex["time"])

    X = ex[["X0", "X1", "X2"]].to_numpy()
    pca = PCA(n_components=2)
    X2d = pca.fit_transform(X)

    # color by time progression (not labels)
    c = np.linspace(0, 1, len(ex))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(X2d[:, 0], X2d[:, 1], s=4, c=c, alpha=0.6)
    ax.set_title(f"Delay embedding projected to 2D (PCA): {example_mouse}  |  τ={TAU_HOURS}h, m={EMBEDDING_DIM}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.3)

    # Show variance explained (nice for presentations)
    ve = pca.explained_variance_ratio_
    ax.text(0.02, 0.98,
            f"Explained var: PC1={ve[0]:.2f}, PC2={ve[1]:.2f}",
            transform=ax.transAxes, va='top')

    plt.tight_layout()
    plt.savefig("embedding_pca2d_example.png", dpi=200)
    plt.close()
    print("Saved: embedding_pca2d_example.png")


