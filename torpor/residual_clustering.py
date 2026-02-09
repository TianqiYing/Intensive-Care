"""
UNSUPERVISED CLUSTERING WITH DYNAMIC RESIDUAL TEMPERATURE
==========================================================
Uses rolling 24-hour baseline (like colleague's SVM approach)

Key change: T_baseline is now DYNAMIC (changes over time)
- Rolling 24h median using only PAST data (closed='left')
- Adapts to drift and baseline changes
- More accurate residual temperature

Feature 1: T_residual = T - T_baseline_dynamic
Feature 2: dT/dt (rate of change)
Feature 3: variability of dT/dt
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')

print("=" * 80)
print("UNSUPERVISED CLUSTERING - DYNAMIC RESIDUAL TEMPERATURE")
print("=" * 80)

# ============================================================================
# CONFIGURATION
# ============================================================================

FASTING_DATE = pd.Timestamp("2023-09-05")
N_CLUSTERS = 4
BASELINE_WINDOW_HOURS = 24  # Rolling window for baseline
MIN_PERIODS_POINTS = 12  # Minimum points for rolling calculation


# ============================================================================
# LOAD AND PREPROCESS
# ============================================================================

def load_and_preprocess(filepath, drop_first_days=3, max_gap_hours=24):
    """Load and preprocess temperature data."""
    # Load
    df = pd.read_excel(filepath)
    df = df[["Time", "Temperature"]].copy()
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df["Temperature"] = pd.to_numeric(df["Temperature"], errors="coerce")
    df = df.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)

    # Detect and truncate at rollover
    df["time_gap_hours"] = df["Time"].diff().dt.total_seconds() / 3600
    rollover_idx = df.index[df["time_gap_hours"] > max_gap_hours]
    if len(rollover_idx) > 0:
        df = df.iloc[:rollover_idx[0]].reset_index(drop=True)

    # Drop first N days (acclimation)
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
        win = max(3, int(round(60 / step)))  # 60-min window

        # Smoothed dT/dt
        df["roc_smooth"] = roc.rolling(win, min_periods=12, center=True).median()

        # Variability of dT/dt
        df["roc_sd"] = roc.rolling(win, min_periods=12, center=True).std()
    else:
        df["roc_smooth"] = np.nan
        df["roc_sd"] = np.nan

    return df


def compute_dynamic_baseline(df, window_hours=24, min_periods=12):
    """
    Compute dynamic baseline using rolling median.

    Key features:
    - Uses ONLY past data (closed='left') - real-time constraint
    - Rolling 24-hour window
    - Falls back to expanding median for early points

    This matches the colleague's SVM approach!

    Parameters:
    -----------
    df : DataFrame
        Must have 'Time' (datetime index) and 'Temperature' columns
    window_hours : int
        Rolling window size in hours (default: 24)
    min_periods : int
        Minimum number of points for rolling calculation

    Returns:
    --------
    df : DataFrame
        With added 'T_baseline_dyn' and 'T_residual' columns
    """
    df = df.copy()

    # Set Time as index for rolling operations
    df_indexed = df.set_index('Time').sort_index()

    # CRITICAL: Rolling baseline using ONLY PAST data
    # closed='left' means each point only sees data BEFORE it
    # This is real-time: no future information leakage!
    df_indexed["T_baseline_dyn"] = df_indexed["Temperature"].rolling(
        window=f"{window_hours}h",
        min_periods=min_periods,
        closed='left'  # ← KEY: Only past data!
    ).median()

    # Early portion fallback: Use expanding median
    # (First 24 hours don't have enough past data)
    if df_indexed["T_baseline_dyn"].isna().any():
        expanding_baseline = df_indexed["Temperature"].expanding(min_periods=1).median()
        df_indexed["T_baseline_dyn"] = df_indexed["T_baseline_dyn"].fillna(expanding_baseline)

    # Compute residual temperature
    df_indexed["T_residual"] = df_indexed["Temperature"] - df_indexed["T_baseline_dyn"]

    # Reset index
    df_result = df_indexed.reset_index()

    return df_result


print("\nSTEP 1: Loading and preprocessing data")
print("-" * 80)

all_data = {}
mice = [f'N{i}' for i in range(1, 17)]

for mouse in mice:
    try:
        df = load_and_preprocess(f'{mouse}.xlsx')

        # Compute DYNAMIC baseline (rolling 24h median)
        df = compute_dynamic_baseline(df, window_hours=BASELINE_WINDOW_HOURS,
                                      min_periods=MIN_PERIODS_POINTS)

        all_data[mouse] = df

        # Summary stats
        baseline_mean = df['T_baseline_dyn'].mean()
        baseline_std = df['T_baseline_dyn'].std()
        residual_min = df['T_residual'].min()
        residual_max = df['T_residual'].max()

        print(f"  {mouse}: {len(df):6d} points | "
              f"Baseline: {baseline_mean:.1f}±{baseline_std:.1f}°C | "
              f"Residual: {residual_min:.1f} to {residual_max:.1f}°C")
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
    # Resample to regular 5-min intervals
    df_5min = df.set_index('Time').resample('5min').agg({
        'Temperature': 'mean',
        'T_baseline_dyn': 'mean',
        'T_residual': 'mean',
        'roc_smooth': 'mean',
        'roc_sd': 'mean'
    }).reset_index()

    df_5min = df_5min.dropna()
    df_5min['mouse'] = mouse
    all_windows.append(df_5min)

windows = pd.concat(all_windows, ignore_index=True)
print(f"Total windows: {len(windows):,}")

# ============================================================================
# UNSUPERVISED CLUSTERING ON DYNAMIC RESIDUAL FEATURES
# ============================================================================

print("\nSTEP 3: Unsupervised K-means clustering on dynamic residual features")
print("-" * 80)

# Phase-space features: DYNAMIC residual temperature, dT/dt, variability
features = ['T_residual', 'roc_smooth', 'roc_sd']
X = windows[features].values

# Remove NaNs
mask = ~np.isnan(X).any(axis=1)
X_clean = X[mask]
windows_clean = windows[mask].copy().reset_index(drop=True)

print(f"Clean windows: {len(windows_clean):,}")
print(f"\nFeatures:")
print(f"  1. T_residual (°C) - DYNAMIC deviation from rolling 24h baseline")
print(f"  2. roc_smooth (°C/min) - cooling vs warming")
print(f"  3. roc_sd (°C/min) - variability of gradient")

# Standardize features
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
    print(f"  T_residual: {cdata['T_residual'].mean():.1f} ± {cdata['T_residual'].std():.1f}°C")
    print(f"  Temperature: {cdata['Temperature'].mean():.1f} ± {cdata['Temperature'].std():.1f}°C")
    print(f"  dT/dt: {cdata['roc_smooth'].mean():.4f} ± {cdata['roc_smooth'].std():.4f}°C/min")

# Torpor = most negative residual temperature
cluster_residuals = windows_clean.groupby('cluster')['T_residual'].mean()
TORPOR_CLUSTER = cluster_residuals.idxmin()

print(f"\n{'=' * 80}")
print(f"TORPOR CLUSTER: {TORPOR_CLUSTER} (T_residual: {cluster_residuals[TORPOR_CLUSTER]:.1f}°C)")
print(f"{'=' * 80}")

# ============================================================================
# CREATE FIGURE - TIME SERIES WITH DYNAMIC RESIDUAL TEMPERATURE
# ============================================================================

print("\nSTEP 5: Creating figure")
print("-" * 80)

CLUSTER_COLORS = {
    0: '#87CEEB',  # Sky Blue
    1: '#FF0000',  # RED
    2: '#90EE90',  # Light Green
    3: '#4682B4',  # Steel Blue
}

# Ensure torpor cluster is red
if TORPOR_CLUSTER != 1:
    temp_color = CLUSTER_COLORS[1]
    CLUSTER_COLORS[1] = CLUSTER_COLORS[TORPOR_CLUSTER]
    CLUSTER_COLORS[TORPOR_CLUSTER] = temp_color

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

    # Plot each cluster (showing RESIDUAL temperature)
    for cid in sorted(mouse_data['cluster'].unique()):
        cdata = mouse_data[mouse_data['cluster'] == cid]

        if cid == TORPOR_CLUSTER:
            # TORPOR cluster - red, prominent
            ax.scatter(cdata['Time'], cdata['T_residual'],
                       c=CLUSTER_COLORS[cid], s=15, alpha=0.95, zorder=10,
                       label=f'Cluster {cid}' if idx == 0 else "",
                       edgecolors='darkred', linewidths=0.5)
        else:
            # Normal clusters - smaller, transparent
            ax.scatter(cdata['Time'], cdata['T_residual'],
                       c=CLUSTER_COLORS[cid], s=5, alpha=0.6,
                       label=f'Cluster {cid}' if idx == 0 else "")

    # Fasting date line
    ax.axvline(FASTING_DATE, color='purple', linestyle='--',
               linewidth=2, alpha=0.7, label='Fasting' if idx == 0 else "", zorder=5)

    # Zero line (baseline)
    ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3, zorder=1)

    # Calculate torpor stats
    n_torpor = (mouse_data['cluster'] == TORPOR_CLUSTER).sum()
    torpor_hours = n_torpor * 5 / 60

    # Y-axis label showing baseline is DYNAMIC
    baseline_mean = mouse_data['T_baseline_dyn'].mean()
    ax.set_ylabel(f'{mouse}\n(Δ°C from\ndynamic\nbaseline)',
                  fontsize=9, fontweight='bold')
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.set_ylim(-15, 5)

    # Add annotation
    if n_torpor > 0:
        torpor_windows = mouse_data[mouse_data['cluster'] == TORPOR_CLUSTER]
        first_torpor = torpor_windows['Time'].min()
        hours_after = (first_torpor - FASTING_DATE).total_seconds() / 3600
        min_residual = torpor_windows['T_residual'].min()

        ax.text(0.01, 0.98,
                f'Cluster {TORPOR_CLUSTER}: {torpor_hours:.1f}h\n'
                f'Starts +{hours_after:.0f}h\n'
                f'Min: {min_residual:.1f}°C below baseline',
                transform=ax.transAxes, fontsize=8, va='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='red', alpha=0.4))
    else:
        ax.text(0.01, 0.98, f'No Cluster {TORPOR_CLUSTER}',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgray', alpha=0.7))

    # Highlight N13 (unfasted control)
    if mouse == 'N13':
        ax.text(0.99, 0.98, 'UNFASTED CONTROL',
                transform=ax.transAxes, fontsize=10, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow',
                          edgecolor='black', linewidth=2), fontweight='bold')

# Legend and labels
axes[0].legend(loc='upper right', fontsize=10, ncol=5, framealpha=0.95)
axes[-1].set_xlabel('Date', fontsize=13, fontweight='bold')
axes[-1].tick_params(axis='x', rotation=45, labelsize=10)

plt.suptitle(f'Dynamic Residual Clustering: Cluster {TORPOR_CLUSTER} (RED) = Torpor\n'
             f'All Mice N1-N16 | K-Means (k={N_CLUSTERS}) | Rolling 24h Baseline',
             fontsize=16, fontweight='bold', y=0.9995)

plt.tight_layout()
fig.savefig('DYNAMIC_RESIDUAL_clustering.png', dpi=150, bbox_inches='tight')
plt.close()

print("Saved: DYNAMIC_RESIDUAL_clustering.png")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\nSTEP 6: Saving results")
print("-" * 80)

# Save complete cluster assignments
windows_clean.to_excel('dynamic_residual_clusters.xlsx', index=False)
print("Saved: dynamic_residual_clusters.xlsx")

# Save cluster statistics
cluster_stats = []
for cid in sorted(windows_clean['cluster'].unique()):
    cdata = windows_clean[windows_clean['cluster'] == cid]
    cluster_stats.append({
        'cluster': cid,
        'n_windows': len(cdata),
        'pct_data': 100 * len(cdata) / len(windows_clean),
        'T_residual_mean': cdata['T_residual'].mean(),
        'T_residual_std': cdata['T_residual'].std(),
        'temp_mean': cdata['Temperature'].mean(),
        'temp_std': cdata['Temperature'].std(),
        'roc_mean': cdata['roc_smooth'].mean(),
        'roc_std': cdata['roc_smooth'].std(),
    })

stats_df = pd.DataFrame(cluster_stats)
stats_df.to_excel('dynamic_residual_statistics.xlsx', index=False)
print("Saved: dynamic_residual_statistics.xlsx")

# Save torpor summary
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
            'baseline_mean': mouse_data['T_baseline_dyn'].mean(),
            'baseline_std': mouse_data['T_baseline_dyn'].std(),
            'n_windows': n_torpor,
            'duration_hours': torpor_hours,
            'onset_hours_post_fasting': hours_after,
            'T_residual_min': torpor_windows['T_residual'].min(),
            'T_residual_mean': torpor_windows['T_residual'].mean(),
            'temp_min': torpor_windows['Temperature'].min(),
            'temp_mean': torpor_windows['Temperature'].mean(),
        })

if torpor_summary:
    torpor_df = pd.DataFrame(torpor_summary)
    torpor_df.to_excel('dynamic_torpor_summary.xlsx', index=False)
    print("Saved: dynamic_torpor_summary.xlsx")

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
print(f"  • T_residual: {cluster_residuals[TORPOR_CLUSTER]:.1f}°C (DYNAMIC deviation from baseline)")

if torpor_summary:
    print(f"  • {len(torpor_summary)} mice show torpor")
    print(f"  • Mean onset: {torpor_df['onset_hours_post_fasting'].mean():.1f}h post-fasting")
    print(f"  • Mean drop: {torpor_df['T_residual_mean'].mean():.1f}°C below dynamic baseline")

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
print(f"  • DYNAMIC_RESIDUAL_clustering.png")
print(f"  • dynamic_residual_clusters.xlsx")
print(f"  • dynamic_residual_statistics.xlsx")
print(f"  • dynamic_torpor_summary.xlsx")

print("\n" + "=" * 80)
print("KEY ADVANTAGES OF DYNAMIC BASELINE:")
print("  • Adapts to baseline drift over time")
print("  • Uses only PAST data (real-time, no future leakage)")
print("  • Matches colleague's SVM approach")
print("  • More accurate residuals for each time point")
print("  • Baseline changes: mean ± std shown in torpor summary")
print("=" * 80)