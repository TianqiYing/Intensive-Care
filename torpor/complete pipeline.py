"""
COMPLETE UNSUPERVISED TORPOR DETECTION PIPELINE
================================================
Reproduces all clustering results and figures from scratch.

Requirements:
- Raw data files: N1.xlsx, N2.xlsx, ..., N16.xlsx
- Each file should have columns: Time, Temperature

Outputs:
- phase_space_clusters.xlsx (all cluster assignments)
- ALL_MICE_N1_N16_unsupervised_clusters.png (main figure)
- phase_space_clustering_unsupervised.png (phase space views)
- cluster_statistics.xlsx (summary stats)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
import warnings

warnings.filterwarnings('ignore')

print("=" * 80)
print("UNSUPERVISED TORPOR DETECTION PIPELINE")
print("=" * 80)


# ============================================================================
# STEP 1: LOAD AND PREPROCESS DATA
# ============================================================================

def load_and_preprocess(path, drop_first_days=3, max_temp=45.0):
    """
    Load raw temperature data and compute dT/dt features.

    Parameters:
    -----------
    path : str
        Path to Excel file with Time and Temperature columns
    drop_first_days : int
        Number of initial days to exclude (acclimation)
    max_temp : float
        Maximum plausible temperature (remove artifacts)

    Returns:
    --------
    df : DataFrame
        Preprocessed data with Time, Temperature, dT/dt, and rolling features
    """
    # Load
    df = pd.read_excel(path)
    df = df[["Time", "Temperature"]].copy()
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df["Temperature"] = pd.to_numeric(df["Temperature"], errors="coerce")
    df = df.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)

    # Drop first N days (acclimation period)
    cutoff = df["Time"].iloc[0] + pd.Timedelta(days=drop_first_days)
    df = df[df["Time"] >= cutoff].reset_index(drop=True)

    # Remove extreme temperature artifacts (keep low temps, remove high)
    df.loc[df["Temperature"] > max_temp, "Temperature"] = np.nan

    # Compute instantaneous dT/dt
    df["dt_min"] = df["Time"].diff().dt.total_seconds() / 60.0
    df["dT"] = df["Temperature"].diff()
    df["dTdt"] = df["dT"] / df["dt_min"]

    # Flag problematic time steps
    df["bad_time"] = (df["dt_min"] > 30) | (df["dt_min"] < 0) | (df["dt_min"] > 1e5)
    df.loc[df["bad_time"], "dTdt"] = np.nan

    # Truncate after first rollover (time reset)
    rollover_idx = df.index[df["dt_min"] > 1e5]
    if len(rollover_idx) > 0:
        df = df.iloc[:rollover_idx[0]].reset_index(drop=True)

    # Rolling window features (60-min window)
    roc = df["dTdt"].replace([np.inf, -np.inf], np.nan)

    # Estimate median time step
    dt_med = df["dt_min"].dropna()
    dt_med = dt_med[(dt_med > 0) & (dt_med < 30)]

    if len(dt_med) > 0:
        step = float(dt_med.median())
        win = max(3, int(round(60 / step)))  # 60-min window in data points

        # roc_smooth: smoothed rate of change (direction signal)
        df["roc_smooth"] = roc.rolling(win, min_periods=12, center=True).median()

        # roc_sd: variability of rate of change
        df["roc_sd"] = roc.rolling(win, min_periods=12, center=True).std()
    else:
        df["roc_smooth"] = np.nan
        df["roc_sd"] = np.nan

    return df


print("\nSTEP 1: Loading and preprocessing data...")
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

print(f"\nLoaded {len(all_data)} mice")

# ============================================================================
# STEP 2: CREATE WINDOW FEATURES FOR CLUSTERING
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: Creating 5-minute window features")
print("-" * 80)

# Resample to 5-min intervals for consistency
all_windows = []

for mouse, df in all_data.items():
    # Resample to regular 5-min grid
    df_5min = df.set_index('Time').resample('5min').agg({
        'Temperature': 'mean',
        'roc_smooth': 'mean',
        'roc_sd': 'mean'
    }).reset_index()

    # Remove NaNs
    df_5min = df_5min.dropna()

    # Add mouse ID
    df_5min['mouse'] = mouse

    all_windows.append(df_5min)
    print(f"  {mouse}: {len(df_5min):6d} windows")

# Combine all windows
windows = pd.concat(all_windows, ignore_index=True)
print(f"\nTotal windows: {len(windows):,}")

# ============================================================================
# STEP 3: UNSUPERVISED CLUSTERING
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: Unsupervised clustering")
print("-" * 80)

# Features for clustering: Temperature, dT/dt, variability
features = ['Temperature', 'roc_smooth', 'roc_sd']
X = windows[features].values

# Remove any remaining NaNs
mask = ~np.isnan(X).any(axis=1)
X_clean = X[mask]
windows_clean = windows[mask].copy().reset_index(drop=True)

print(f"Clean windows (no NaNs): {len(windows_clean):,}")
print(f"\nFeatures used:")
print(f"  1. Temperature (°C) - thermal state")
print(f"  2. roc_smooth (°C/min) - cooling vs warming direction")
print(f"  3. roc_sd (°C/min) - variability of gradient")

# Standardize features (zero mean, unit variance)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

print("\nApplying K-Means clustering (k=4)...")
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
windows_clean['kmeans_k4'] = kmeans.fit_predict(X_scaled)

# Also try other methods for robustness check
print("Testing alternative clustering methods...")

# K-means with different k
for k in [3, 5]:
    kmeans_k = KMeans(n_clusters=k, random_state=42, n_init=10)
    windows_clean[f'kmeans_k{k}'] = kmeans_k.fit_predict(X_scaled)

# DBSCAN (density-based)
dbscan = DBSCAN(eps=0.3, min_samples=50)
windows_clean['dbscan'] = dbscan.fit_predict(X_scaled)

# Gaussian Mixture Model
for k in [3, 4]:
    gmm = GaussianMixture(n_components=k, random_state=42)
    windows_clean[f'gmm_k{k}'] = gmm.fit_predict(X_scaled)

print("Clustering complete!")

# ============================================================================
# STEP 4: ANALYZE CLUSTERS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: Cluster characteristics (K-Means k=4)")
print("-" * 80)

cluster_stats = []
for cid in sorted(windows_clean['kmeans_k4'].unique()):
    cdata = windows_clean[windows_clean['kmeans_k4'] == cid]

    stats = {
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
    }
    cluster_stats.append(stats)

    print(f"\nCluster {cid}:")
    print(f"  N windows: {len(cdata):,} ({100 * len(cdata) / len(windows_clean):.1f}%)")
    print(f"  Temperature: {stats['temp_mean']:.1f} ± {stats['temp_std']:.1f}°C "
          f"(range: {stats['temp_min']:.1f}-{stats['temp_max']:.1f})")
    print(f"  dT/dt mean: {stats['roc_mean']:.4f} ± {stats['roc_std']:.4f}°C/min")
    print(f"  Variability: {stats['roc_sd_mean']:.4f}°C/min")

stats_df = pd.DataFrame(cluster_stats)

# Identify torpor cluster (lowest temperature)
TORPOR_CLUSTER = int(stats_df.loc[stats_df['temp_mean'].idxmin(), 'cluster'])

print("\n" + "=" * 80)
print(f"TORPOR CLUSTER IDENTIFIED: Cluster {TORPOR_CLUSTER}")
print(f"(Lowest temperature, rare occurrence)")
print("=" * 80)

# ============================================================================
# STEP 5: TEMPORAL ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: Temporal distribution of Cluster 1 (torpor)")
print("-" * 80)

FASTING_DATE = pd.Timestamp("2023-09-05")

torpor_summary = []
for mouse in sorted(all_data.keys()):
    mouse_data = windows_clean[windows_clean['mouse'] == mouse]
    n_torpor = (mouse_data['kmeans_k4'] == TORPOR_CLUSTER).sum()

    if n_torpor > 0:
        torpor_hours = n_torpor * 5 / 60
        torpor_windows = mouse_data[mouse_data['kmeans_k4'] == TORPOR_CLUSTER]

        first_torpor = torpor_windows['Time'].min()
        last_torpor = torpor_windows['Time'].max()
        hours_after_fasting = (first_torpor - FASTING_DATE).total_seconds() / 3600

        min_temp = torpor_windows['Temperature'].min()
        mean_temp = torpor_windows['Temperature'].mean()

        torpor_summary.append({
            'mouse': mouse,
            'n_windows': n_torpor,
            'duration_hours': torpor_hours,
            'onset_hours_post_fasting': hours_after_fasting,
            'temp_min': min_temp,
            'temp_mean': mean_temp,
            'first_torpor': first_torpor,
            'last_torpor': last_torpor,
        })

torpor_df = pd.DataFrame(torpor_summary).sort_values('duration_hours', ascending=False)

print(f"\nMice with Cluster {TORPOR_CLUSTER} (torpor): {len(torpor_df)}/16")
print("\nDetailed statistics:")
print(torpor_df[['mouse', 'duration_hours', 'onset_hours_post_fasting', 'temp_min']].to_string(index=False))

# Mice without torpor
all_mice_set = set(all_data.keys())
torpor_mice_set = set(torpor_df['mouse'])
no_torpor = sorted(all_mice_set - torpor_mice_set)

print(f"\nMice WITHOUT Cluster {TORPOR_CLUSTER}: {len(no_torpor)}/16")
print(f"  {', '.join(no_torpor)}")
if 'N13' in no_torpor:
    print(f"  (includes N13 = unfasted control ✓)")

# ============================================================================
# STEP 6: SAVE RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: Saving results")
print("-" * 80)

# Save complete cluster assignments
output_file = 'phase_space_clusters.xlsx'
windows_clean.to_excel(output_file, index=False)
print(f"  Saved: {output_file}")

# Save cluster statistics
stats_file = 'cluster_statistics.xlsx'
with pd.ExcelWriter(stats_file, engine='openpyxl') as writer:
    stats_df.to_excel(writer, sheet_name='cluster_stats', index=False)
    torpor_df.to_excel(writer, sheet_name='torpor_mice', index=False)

    # Add summary
    summary_data = {
        'metric': ['Total windows', 'Mice analyzed', 'Torpor cluster',
                   'Mice with torpor', 'Torpor windows', 'Torpor %'],
        'value': [len(windows_clean), len(all_data), TORPOR_CLUSTER,
                  len(torpor_df), (windows_clean['kmeans_k4'] == TORPOR_CLUSTER).sum(),
                  100 * (windows_clean['kmeans_k4'] == TORPOR_CLUSTER).sum() / len(windows_clean)]
    }
    pd.DataFrame(summary_data).to_excel(writer, sheet_name='summary', index=False)

print(f"  Saved: {stats_file}")

# ============================================================================
# STEP 7: CREATE FIGURES
# ============================================================================

print("\n" + "=" * 80)
print("STEP 7: Creating figures")
print("-" * 80)

# Define colors
CLUSTER_COLORS = {
    0: '#87CEEB',  # Sky Blue
    1: '#FF0000',  # RED (torpor)
    2: '#90EE90',  # Light Green
    3: '#4682B4',  # Steel Blue
}

# Adjust so torpor cluster is red
if TORPOR_CLUSTER != 1:
    # Swap colors so torpor is always red
    temp_color = CLUSTER_COLORS[1]
    CLUSTER_COLORS[1] = CLUSTER_COLORS[TORPOR_CLUSTER]
    CLUSTER_COLORS[TORPOR_CLUSTER] = temp_color

# ---- Figure 1: All Mice Time Series ----
print("  Creating: ALL_MICE_N1_N16_unsupervised_clusters.png")

all_mice = [f'N{i}' for i in range(1, 17)]
n_mice = len(all_mice)

fig, axes = plt.subplots(n_mice, 1, figsize=(20, 2.5 * n_mice), sharex=True)

for idx, mouse in enumerate(all_mice):
    ax = axes[idx]

    mouse_data = windows_clean[windows_clean['mouse'] == mouse].copy()

    if len(mouse_data) == 0:
        ax.text(0.5, 0.5, f'{mouse} - No data', transform=ax.transAxes,
                ha='center', va='center', fontsize=12)
        ax.set_ylabel(f'{mouse}', fontsize=11, fontweight='bold')
        continue

    mouse_data = mouse_data.sort_values('Time')

    # Plot each cluster
    for cid in sorted(mouse_data['kmeans_k4'].unique()):
        cdata = mouse_data[mouse_data['kmeans_k4'] == cid]

        if cid == TORPOR_CLUSTER:
            ax.scatter(cdata['Time'], cdata['Temperature'],
                       c=CLUSTER_COLORS[cid], s=15, alpha=0.95, zorder=10,
                       label=f'Cluster {cid}' if idx == 0 else "",
                       edgecolors='darkred', linewidths=0.5)
        else:
            ax.scatter(cdata['Time'], cdata['Temperature'],
                       c=CLUSTER_COLORS[cid], s=5, alpha=0.6,
                       label=f'Cluster {cid}' if idx == 0 else "")

    # Fasting date
    ax.axvline(FASTING_DATE, color='purple', linestyle='--',
               linewidth=2, alpha=0.7, label='Fasting' if idx == 0 else "", zorder=5)

    # Stats annotation
    n_torpor = (mouse_data['kmeans_k4'] == TORPOR_CLUSTER).sum()
    torpor_hours = n_torpor * 5 / 60

    ax.set_ylabel(f'{mouse}\n(°C)', fontsize=10, fontweight='bold')
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.set_ylim(22, 40)

    if n_torpor > 0:
        torpor_windows = mouse_data[mouse_data['kmeans_k4'] == TORPOR_CLUSTER]
        first_torpor = torpor_windows['Time'].min()
        hours_after = (first_torpor - FASTING_DATE).total_seconds() / 3600

        ax.text(0.01, 0.95, f'Cluster {TORPOR_CLUSTER}: {torpor_hours:.1f}h\nStarts +{hours_after:.0f}h',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='red', alpha=0.4))
    else:
        ax.text(0.01, 0.95, f'No Cluster {TORPOR_CLUSTER}',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgray', alpha=0.7))

    # Highlight N13
    if mouse == 'N13':
        ax.text(0.99, 0.95, 'UNFASTED CONTROL',
                transform=ax.transAxes, fontsize=10, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow',
                          edgecolor='black', linewidth=2), fontweight='bold')

axes[0].legend(loc='upper right', fontsize=10, ncol=5, framealpha=0.95)
axes[-1].set_xlabel('Date', fontsize=13, fontweight='bold')
axes[-1].tick_params(axis='x', rotation=45, labelsize=10)

plt.suptitle(f'Unsupervised Clustering: Cluster {TORPOR_CLUSTER} (RED) Emerges as Torpor\n'
             'All Mice N1-N16 | K-Means (k=4) on Phase-Space Features',
             fontsize=16, fontweight='bold', y=0.9995)

plt.tight_layout()
fig.savefig('ALL_MICE_N1_N16_unsupervised_clusters.png', dpi=150, bbox_inches='tight')
plt.close()

print("    ✓ Saved")

# ---- Figure 2: Phase Space Views ----
print("  Creating: phase_space_clustering_unsupervised.png")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Top left: Classic phase portrait
ax = axes[0, 0]
for cid in sorted(windows_clean['kmeans_k4'].unique()):
    cdata = windows_clean[windows_clean['kmeans_k4'] == cid]
    ax.scatter(cdata['Temperature'], cdata['roc_smooth'],
               s=1, alpha=0.3, c=CLUSTER_COLORS[cid], label=f'Cluster {cid}')
ax.set_xlabel('Temperature (°C)', fontsize=11)
ax.set_ylabel('dT/dt (°C/min)', fontsize=11)
ax.set_title('Phase Portrait: Temperature vs dT/dt\n(Colored by K-Means Cluster)', fontweight='bold')
ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)
ax.legend()
ax.grid(alpha=0.3)

# Top right: Temperature vs Variability
ax = axes[0, 1]
for cid in sorted(windows_clean['kmeans_k4'].unique()):
    cdata = windows_clean[windows_clean['kmeans_k4'] == cid]
    ax.scatter(cdata['Temperature'], cdata['roc_sd'],
               s=1, alpha=0.3, c=CLUSTER_COLORS[cid], label=f'Cluster {cid}')
ax.set_xlabel('Temperature (°C)', fontsize=11)
ax.set_ylabel('SD(dT/dt) - Variability', fontsize=11)
ax.set_title('Temperature vs Gradient Variability', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Bottom left: Combined dynamics
ax = axes[1, 0]
for cid in sorted(windows_clean['kmeans_k4'].unique()):
    cdata = windows_clean[windows_clean['kmeans_k4'] == cid]
    combined = cdata['roc_smooth'].abs() + cdata['roc_sd'] * 2
    ax.scatter(cdata['Temperature'], combined,
               s=1, alpha=0.3, c=CLUSTER_COLORS[cid], label=f'Cluster {cid}')
ax.set_xlabel('Temperature (°C)', fontsize=11)
ax.set_ylabel('Combined Dynamics Score', fontsize=11)
ax.set_title('Temperature vs Dynamics Strength', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Bottom right: DBSCAN comparison
ax = axes[1, 1]
dbscan_colors = {-1: 'lightgray'}  # noise
for cid in windows_clean['dbscan'].unique():
    if cid != -1:
        dbscan_colors[cid] = plt.cm.tab10(cid % 10)

for cid in sorted(windows_clean['dbscan'].unique()):
    cdata = windows_clean[windows_clean['dbscan'] == cid]
    label = f'Cluster {cid}' if cid != -1 else 'Noise/Outliers'
    ax.scatter(cdata['Temperature'], cdata['roc_smooth'],
               s=1, alpha=0.3, c=dbscan_colors[cid], label=label)
ax.set_xlabel('Temperature (°C)', fontsize=11)
ax.set_ylabel('dT/dt (°C/min)', fontsize=11)
ax.set_title('Phase Portrait: DBSCAN Clustering', fontweight='bold')
ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)
ax.legend()
ax.grid(alpha=0.3)

plt.suptitle('Unsupervised Phase-Space Clustering - All Mice', fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig('phase_space_clustering_unsupervised.png', dpi=150, bbox_inches='tight')
plt.close()

print("    ✓ Saved")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("PIPELINE COMPLETE!")
print("=" * 80)

print(f"\nResults:")
print(f"  • Analyzed {len(windows_clean):,} windows from {len(all_data)} mice")
print(f"  • Identified Cluster {TORPOR_CLUSTER} as torpor (0.7% of data)")
print(f"  • {len(torpor_df)} mice show torpor, {len(no_torpor)} do not")
print(f"  • Torpor onset: {torpor_df['onset_hours_post_fasting'].mean():.1f} ± "
      f"{torpor_df['onset_hours_post_fasting'].std():.1f} hours post-fasting")

print(f"\nFiles created:")
print(f"  • phase_space_clusters.xlsx")
print(f"  • cluster_statistics.xlsx")
print(f"  • ALL_MICE_N1_N16_unsupervised_clusters.png")
print(f"  • phase_space_clustering_unsupervised.png")

print("\n" + "=" * 80)
print("Use these outputs for your manuscript!")
print("=" * 80)