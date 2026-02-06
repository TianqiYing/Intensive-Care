"""
UNSUPERVISED TORPOR CLUSTERING - FINAL VERSION
===============================================
Produces clean figure without time gaps.
Handles data rollovers automatically.

Input: N1.xlsx, N2.xlsx, ..., N16.xlsx (Time, Temperature columns)
Output: Clean clustering figure + data files

Run: python clustering_final.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')

print("=" * 80)
print("UNSUPERVISED CLUSTERING - FINAL VERSION")
print("=" * 80)

# ============================================================================
# CONFIGURATION
# ============================================================================

FASTING_DATE = pd.Timestamp("2023-09-05")
N_CLUSTERS = 4


# ============================================================================
# LOAD AND PREPROCESS WITH ROLLOVER FIX
# ============================================================================

def load_and_preprocess(filepath, drop_first_days=3, max_gap_hours=24):
    """
    Load temperature data and compute phase-space features.
    Automatically handles timestamp rollovers by truncating.

    Parameters:
    -----------
    filepath : str
        Path to Excel file with 'Time' and 'Temperature' columns
    drop_first_days : int
        Number of initial days to exclude (acclimation)
    max_gap_hours : float
        If time gap exceeds this, truncate data (fixes rollovers)

    Returns:
    --------
    df : DataFrame
        Preprocessed data with phase-space features
    """
    # Load
    df = pd.read_excel(filepath)
    df = df[["Time", "Temperature"]].copy()
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df["Temperature"] = pd.to_numeric(df["Temperature"], errors="coerce")
    df = df.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)

    # Detect and truncate at rollover (fixes time gaps in visualization)
    df["time_gap_hours"] = df["Time"].diff().dt.total_seconds() / 3600
    rollover_idx = df.index[df["time_gap_hours"] > max_gap_hours]

    if len(rollover_idx) > 0:
        df = df.iloc[:rollover_idx[0]].reset_index(drop=True)

    # Drop first N days (acclimation period)
    cutoff = df["Time"].iloc[0] + pd.Timedelta(days=drop_first_days)
    df = df[df["Time"] >= cutoff].reset_index(drop=True)

    # Remove extreme temperature artifacts
    df.loc[df["Temperature"] > 45.0, "Temperature"] = np.nan

    # Compute instantaneous rate of change (dT/dt)
    df["dt_min"] = df["Time"].diff().dt.total_seconds() / 60.0
    df["dT"] = df["Temperature"].diff()
    df["dTdt"] = df["dT"] / df["dt_min"]

    # Flag problematic time steps
    df["bad_time"] = (df["dt_min"] > 30) | (df["dt_min"] < 0)
    df.loc[df["bad_time"], "dTdt"] = np.nan

    # Compute rolling window features (60-min window)
    roc = df["dTdt"].replace([np.inf, -np.inf], np.nan)

    # Estimate median time step
    dt_med = df["dt_min"].dropna()
    dt_med = dt_med[(dt_med > 0) & (dt_med < 30)]

    if len(dt_med) > 0:
        step = float(dt_med.median())
        win = max(3, int(round(60 / step)))  # 60-min window in samples

        # Smoothed dT/dt (direction: cooling vs warming)
        df["roc_smooth"] = roc.rolling(win, min_periods=12, center=True).median()

        # Variability of dT/dt
        df["roc_sd"] = roc.rolling(win, min_periods=12, center=True).std()
    else:
        df["roc_smooth"] = np.nan
        df["roc_sd"] = np.nan

    return df


print("\nSTEP 1: Loading and preprocessing data")
print("-" * 80)

all_data = {}
mice = [f'N{i}' for i in range(1, 17)]

for mouse in mice:
    try:
        df = load_and_preprocess(f'{mouse}.xlsx')
        all_data[mouse] = df
        print(f"  {mouse}: {len(df):6d} points, {df['Time'].min().date()} to {df['Time'].max().date()}")
    except Exception as e:
        print(f"  {mouse}: FAILED - {e}")

print(f"\nLoaded {len(all_data)} mice successfully")

# ============================================================================
# CREATE 5-MINUTE WINDOW FEATURES
# ============================================================================

print("\nSTEP 2: Creating 5-minute window features")
print("-" * 80)

all_windows = []

for mouse, df in all_data.items():
    # Resample to regular 5-min intervals for consistent windowing
    df_5min = df.set_index('Time').resample('5min').agg({
        'Temperature': 'mean',
        'roc_smooth': 'mean',
        'roc_sd': 'mean'
    }).reset_index()

    df_5min = df_5min.dropna()
    df_5min['mouse'] = mouse
    all_windows.append(df_5min)

windows = pd.concat(all_windows, ignore_index=True)
print(f"Total windows: {len(windows):,}")

# ============================================================================
# UNSUPERVISED CLUSTERING
# ============================================================================

print("\nSTEP 3: Unsupervised K-means clustering")
print("-" * 80)

# Phase-space features: Temperature, dT/dt, variability
features = ['Temperature', 'roc_smooth', 'roc_sd']
X = windows[features].values

# Remove NaNs
mask = ~np.isnan(X).any(axis=1)
X_clean = X[mask]
windows_clean = windows[mask].copy().reset_index(drop=True)

print(f"Clean windows: {len(windows_clean):,}")
print(f"\nFeatures:")
print(f"  1. Temperature (°C) - thermal state")
print(f"  2. roc_smooth (°C/min) - cooling vs warming direction")
print(f"  3. roc_sd (°C/min) - variability of thermal gradient")

# Standardize features (zero mean, unit variance)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

# K-means clustering
print(f"\nApplying K-means (k={N_CLUSTERS})...")
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
windows_clean['cluster'] = kmeans.fit_predict(X_scaled)

# ============================================================================
# IDENTIFY TORPOR CLUSTER
# ============================================================================

print("\nSTEP 4: Analyzing clusters")
print("-" * 80)

for cid in sorted(windows_clean['cluster'].unique()):
    cdata = windows_clean[windows_clean['cluster'] == cid]

    print(f"\nCluster {cid}:")
    print(f"  N windows: {len(cdata):,} ({100 * len(cdata) / len(windows_clean):.1f}%)")
    print(f"  Temperature: {cdata['Temperature'].mean():.1f} ± {cdata['Temperature'].std():.1f}°C")
    print(f"  dT/dt: {cdata['roc_smooth'].mean():.4f} ± {cdata['roc_smooth'].std():.4f}°C/min")
    print(f"  Variability: {cdata['roc_sd'].mean():.4f}°C/min")

# Torpor = lowest temperature cluster
cluster_temps = windows_clean.groupby('cluster')['Temperature'].mean()
TORPOR_CLUSTER = cluster_temps.idxmin()

print(f"\n{'=' * 80}")
print(f"TORPOR CLUSTER: {TORPOR_CLUSTER} (temperature: {cluster_temps[TORPOR_CLUSTER]:.1f}°C)")
print(f"{'=' * 80}")

# ============================================================================
# CREATE FIGURE
# ============================================================================

print("\nSTEP 5: Creating figure")
print("-" * 80)

# Cluster colors
CLUSTER_COLORS = {
    0: '#87CEEB',  # Sky Blue
    1: '#FF0000',  # RED (torpor)
    2: '#90EE90',  # Light Green
    3: '#4682B4',  # Steel Blue
}

fig, axes = plt.subplots(16, 1, figsize=(20, 40), sharex=True)

for idx, mouse in enumerate([f'N{i}' for i in range(1, 17)]):
    ax = axes[idx]

    mouse_data = windows_clean[windows_clean['mouse'] == mouse].copy()

    if len(mouse_data) == 0:
        ax.text(0.5, 0.5, f'{mouse} - No data',
                transform=ax.transAxes, ha='center', va='center', fontsize=12)
        ax.set_ylabel(f'{mouse}', fontsize=11, fontweight='bold')
        continue

    mouse_data = mouse_data.sort_values('Time')

    # Plot each cluster
    for cid in sorted(mouse_data['cluster'].unique()):
        cdata = mouse_data[mouse_data['cluster'] == cid]

        if cid == TORPOR_CLUSTER:
            # TORPOR cluster - red, prominent
            ax.scatter(cdata['Time'], cdata['Temperature'],
                       c=CLUSTER_COLORS[cid], s=15, alpha=0.95, zorder=10,
                       label=f'Cluster {cid}' if idx == 0 else "",
                       edgecolors='darkred', linewidths=0.5)
        else:
            # Normal clusters - smaller, transparent
            ax.scatter(cdata['Time'], cdata['Temperature'],
                       c=CLUSTER_COLORS[cid], s=5, alpha=0.6,
                       label=f'Cluster {cid}' if idx == 0 else "")

    # Fasting date line
    ax.axvline(FASTING_DATE, color='purple', linestyle='--',
               linewidth=2, alpha=0.7, label='Fasting' if idx == 0 else "", zorder=5)

    # Calculate torpor stats
    n_torpor = (mouse_data['cluster'] == TORPOR_CLUSTER).sum()
    torpor_hours = n_torpor * 5 / 60

    ax.set_ylabel(f'{mouse}\n(°C)', fontsize=10, fontweight='bold')
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.set_ylim(22, 40)

    # Add annotation
    if n_torpor > 0:
        torpor_windows = mouse_data[mouse_data['cluster'] == TORPOR_CLUSTER]
        first_torpor = torpor_windows['Time'].min()
        hours_after = (first_torpor - FASTING_DATE).total_seconds() / 3600

        ax.text(0.01, 0.95,
                f'Cluster {TORPOR_CLUSTER}: {torpor_hours:.1f}h\nStarts +{hours_after:.0f}h',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='red', alpha=0.4))
    else:
        ax.text(0.01, 0.95, f'No Cluster {TORPOR_CLUSTER}',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgray', alpha=0.7))

    # Highlight N13 (unfasted control)
    if mouse == 'N13':
        ax.text(0.99, 0.95, 'UNFASTED CONTROL',
                transform=ax.transAxes, fontsize=10, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow',
                          edgecolor='black', linewidth=2), fontweight='bold')

# Legend and labels
axes[0].legend(loc='upper right', fontsize=10, ncol=5, framealpha=0.95)
axes[-1].set_xlabel('Date', fontsize=13, fontweight='bold')
axes[-1].tick_params(axis='x', rotation=45, labelsize=10)

plt.suptitle(f'Unsupervised Clustering: Cluster {TORPOR_CLUSTER} (RED) Emerges as Torpor\n'
             f'All Mice N1-N16 | K-Means (k={N_CLUSTERS}) on Phase-Space Features',
             fontsize=16, fontweight='bold', y=0.9995)

plt.tight_layout()
fig.savefig('ALL_MICE_N1_N16_unsupervised_clusters.png', dpi=150, bbox_inches='tight')
plt.close()

print("Saved: ALL_MICE_N1_N16_unsupervised_clusters.png")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\nSTEP 6: Saving results")
print("-" * 80)

# Save complete cluster assignments
windows_clean.to_excel('phase_space_clusters.xlsx', index=False)
print("Saved: phase_space_clusters.xlsx")

# Save cluster statistics
cluster_stats = []
for cid in sorted(windows_clean['cluster'].unique()):
    cdata = windows_clean[windows_clean['cluster'] == cid]
    cluster_stats.append({
        'cluster': cid,
        'n_windows': len(cdata),
        'pct_data': 100 * len(cdata) / len(windows_clean),
        'temp_mean': cdata['Temperature'].mean(),
        'temp_std': cdata['Temperature'].std(),
        'temp_min': cdata['Temperature'].min(),
        'temp_max': cdata['Temperature'].max(),
        'roc_mean': cdata['roc_smooth'].mean(),
        'roc_std': cdata['roc_smooth'].std(),
        'roc_sd_mean': cdata['roc_sd'].mean(),
    })

stats_df = pd.DataFrame(cluster_stats)
stats_df.to_excel('cluster_statistics.xlsx', index=False)
print("Saved: cluster_statistics.xlsx")

# Save torpor summary per mouse
torpor_summary = []
for mouse in sorted(all_data.keys()):
    mouse_data = windows_clean[windows_clean['mouse'] == mouse]
    n_torpor = (mouse_data['cluster'] == TORPOR_CLUSTER).sum()

    if n_torpor > 0:
        torpor_hours = n_torpor * 5 / 60
        torpor_windows = mouse_data[mouse_data['cluster'] == TORPOR_CLUSTER]
        first_torpor = torpor_windows['Time'].min()
        hours_after = (first_torpor - FASTING_DATE).total_seconds() / 3600

        torpor_summary.append({
            'mouse': mouse,
            'n_windows': n_torpor,
            'duration_hours': torpor_hours,
            'onset_hours_post_fasting': hours_after,
            'temp_min': torpor_windows['Temperature'].min(),
            'temp_mean': torpor_windows['Temperature'].mean(),
        })

if torpor_summary:
    torpor_df = pd.DataFrame(torpor_summary)
    torpor_df.to_excel('torpor_summary.xlsx', index=False)
    print("Saved: torpor_summary.xlsx")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("COMPLETE!")
print("=" * 80)

print(f"\nResults:")
print(f"  • Analyzed {len(windows_clean):,} windows from {len(all_data)} mice")
print(f"  • Identified Cluster {TORPOR_CLUSTER} as torpor")
print(f"  • Cluster {TORPOR_CLUSTER}: {(windows_clean['cluster'] == TORPOR_CLUSTER).sum()} windows "
      f"({100 * (windows_clean['cluster'] == TORPOR_CLUSTER).sum() / len(windows_clean):.1f}%)")
print(f"  • Temperature: {cluster_temps[TORPOR_CLUSTER]:.1f}°C")

if torpor_summary:
    print(f"  • {len(torpor_summary)} mice show torpor")
    print(f"  • Mean onset: {torpor_df['onset_hours_post_fasting'].mean():.1f}h post-fasting")

# Check N13
n13_data = windows_clean[windows_clean['mouse'] == 'N13']
n13_torpor = (n13_data['cluster'] == TORPOR_CLUSTER).sum()
print(f"\nValidation:")
print(f"  • N13 (unfasted control): {n13_torpor} torpor windows", end="")
if n13_torpor == 0:
    print(" ✓")
else:
    print(" ⚠")

print(f"\nGenerated files:")
print(f"  • ALL_MICE_N1_N16_unsupervised_clusters.png")
print(f"  • phase_space_clusters.xlsx")
print(f"  • cluster_statistics.xlsx")
print(f"  • torpor_summary.xlsx")

print("\n" + "=" * 80)
print("Ready for publication!")
print("=" * 80)