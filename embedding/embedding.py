"""
DELAY EMBEDDING (Takens) - TEMPERATURE ONLY - N6
================================================

Goal:
- Create time-delay embedded state vectors for clustering:
    X(t) = [T(t), T(t-τ), T(t-2τ)]

This is NOT supervised and uses no torpor heuristics.
It encodes dynamics/phase geometry.

Outputs:
- N6_delay_embedding_tau360_m3.csv
- N6_temperature_timeseries.png
- N6_delay_embedding_3d.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

print("=" * 70)
print("DELAY EMBEDDING (TEMPERATURE ONLY) - N6")
print("=" * 70)

# ==================================================
# SETTINGS
# ==================================================
FILEPATH = r"C:\Users\harim\PycharmProjects\SVM\N6.xlsx"

CALIB_HOURS = 3
RESAMPLE_TO_1MIN = True

# Delay embedding parameters
TAU_MINUTES = 360   # 6 hours (from 24h/4)
M = 3               # embedding dimension (3D is easy to visualize)

# Optional: standardize temperature (recommended if you later pool mice)
Z_SCORE = True

# Optional: downsample embedded points (keeps file smaller / clustering faster)
# e.g., keep every 5th minute = 5-min step
DOWNSAMPLE_STEP = 5   # set to 1 to keep every minute

# ==================================================
# LOAD + CLEAN
# ==================================================
df = pd.read_excel(FILEPATH)
df["Time"] = pd.to_datetime(df["Time"])
df = df.sort_values("Time").reset_index(drop=True)

print(f"\nLoaded: {len(df)} rows from {FILEPATH}")
print(f"Start: {df['Time'].iloc[0]}")
print(f"End:   {df['Time'].iloc[-1]}")

# Remove calibration
t0 = df["Time"].iloc[0]
cutoff = t0 + pd.Timedelta(hours=CALIB_HOURS)
df = df[df["Time"] >= cutoff].copy()
df = df.reset_index(drop=True)
print(f"\nRemoved first {CALIB_HOURS}h. New start: {df['Time'].iloc[0]}")
print(f"Rows after cut: {len(df)}")

# Resample to strict 1-min grid (important for correct τ in minutes)
if RESAMPLE_TO_1MIN:
    df = df.set_index("Time").resample("1min").mean()
    df["Temperature"] = df["Temperature"].interpolate(limit_direction="both")
    df = df.reset_index()
    print(f"After resampling to 1-min grid: {len(df)} rows")

T = df["Temperature"].to_numpy(dtype=float)
time = df["Time"].to_numpy()

# Optional standardization (helps compare across mice)
if Z_SCORE:
    T_mean = float(np.mean(T))
    T_std = float(np.std(T)) + 1e-12
    Tz = (T - T_mean) / T_std
    print(f"\nZ-scoring enabled: mean={T_mean:.3f}, std={T_std:.3f}")
else:
    Tz = T.copy()

# ==================================================
# BUILD DELAY EMBEDDING
# ==================================================
print("\n" + "=" * 70)
print("BUILDING EMBEDDING")
print("=" * 70)

tau = int(TAU_MINUTES)  # because sampling is 1/min after resampling
if tau <= 0:
    raise ValueError("TAU_MINUTES must be >= 1")
if M < 2:
    raise ValueError("M must be >= 2")

max_lag = (M - 1) * tau
if len(Tz) <= max_lag:
    raise ValueError("Time series too short for chosen tau and M.")

# Create embedded points
# X0 = T(t), X1 = T(t-τ), X2 = T(t-2τ)
# We'll align everything to valid times t where all lags exist.
indices = np.arange(max_lag, len(Tz))

X_cols = []
for k in range(M):
    lag = k * tau
    X_cols.append(Tz[indices - lag])

X = np.vstack(X_cols).T  # shape (N_points, M)
t_embed = df["Time"].iloc[indices].to_numpy()

print(f"Embedding created:")
print(f"  τ = {TAU_MINUTES} minutes ({TAU_MINUTES/60:.1f} h)")
print(f"  m = {M}")
print(f"  Points = {X.shape[0]}")

# Downsample if requested
if DOWNSAMPLE_STEP > 1:
    X = X[::DOWNSAMPLE_STEP]
    t_embed = t_embed[::DOWNSAMPLE_STEP]
    print(f"Downsampled every {DOWNSAMPLE_STEP} minutes -> Points now = {X.shape[0]}")

# ==================================================
# SAVE EMBEDDING CSV
# ==================================================
stem = Path(FILEPATH).stem
out_csv = f"{stem}_delay_embedding_tau{TAU_MINUTES}_m{M}.csv"

out_df = pd.DataFrame({
    "Time": t_embed,
    "X0_T(t)": X[:, 0],
    "X1_T(t-tau)": X[:, 1],
    "X2_T(t-2tau)": X[:, 2],
})

out_df.to_csv(out_csv, index=False)
print(f"\nSaved embedding CSV: {out_csv}")

# ==================================================
# PLOTS
# ==================================================
print("\n" + "=" * 70)
print("PLOTTING")
print("=" * 70)

# Plot 1: temperature time series (raw)
plt.figure(figsize=(14, 4))
plt.plot(df["Time"], T, linewidth=0.6)
plt.ylabel("Temperature (°C)")
plt.title(f"{stem} Temperature Time Series (post-calibration)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
ts_png = f"{stem}_temperature_timeseries.png"
plt.savefig(ts_png, dpi=300)
plt.close()
print(f"Saved: {ts_png}")

# Plot 2: 3D embedding scatter
# We'll color by time progression (no labels)
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# color index: 0..1 across time
c = np.linspace(0, 1, len(X))

ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=2, c=c)
ax.set_xlabel("T(t) (z-scored)" if Z_SCORE else "T(t)")
ax.set_ylabel("T(t-τ)")
ax.set_zlabel("T(t-2τ)")
ax.set_title(f"{stem} Delay Embedding (τ={TAU_MINUTES}min, m={M})")

plt.tight_layout()
emb_png = f"{stem}_delay_embedding_3d.png"
plt.savefig(emb_png, dpi=300)
plt.close()
print(f"Saved: {emb_png}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print(f"""
What you now have:
- A point cloud in 3D state space: [T(t), T(t-τ), T(t-2τ)]
- This is a 'time embedding' your colleague can cluster without labels.

How to use for clustering:
- Cluster the rows of the CSV (X0,X1,X2)
- Clusters correspond to regions of the reconstructed attractor
- Transitions are trajectories between clusters

Next simple validations:
- Try τ = 300, 360, 420 (5h, 6h, 7h) and see which gives cleanest geometry
- Try m = 4 later if needed (but 3 is easiest to start)
""")
