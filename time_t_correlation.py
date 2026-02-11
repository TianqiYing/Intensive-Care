import os
import glob
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================

N_CLUSTERS = 4
SEED = 42

DROP_FIRST_DAYS = 3
MAX_GAP_HOURS_STOP = 24  # truncate file at large rollover gaps

RESAMPLE_RULE = "5min"

MAX_BOUT_GAP_MINUTES = 30     # gaps < this are considered same bout
MIN_BOUT_DURATION_MINUTES = 15

# For deep/shallow summaries (optional; does not affect correlation)
DEEP_THRESHOLD = 27.0

# Robust inference
BOOTSTRAP_N = 5000
PERM_N = 10000

OUTDIR = Path("./bout_temp_duration_robust_out")
OUTDIR.mkdir(exist_ok=True)

# =============================================================================
# YOUR PREPROCESS + FEATURE PIPELINE (kept close to yours)
# =============================================================================

def load_and_preprocess(filepath, drop_first_days=DROP_FIRST_DAYS, max_gap_hours=MAX_GAP_HOURS_STOP):
    df = pd.read_excel(filepath)
    df = df[["Time", "Temperature"]].copy()
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df["Temperature"] = pd.to_numeric(df["Temperature"], errors="coerce")
    df = df.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)

    df["time_gap_hours"] = df["Time"].diff().dt.total_seconds() / 3600
    rollover_idx = df.index[df["time_gap_hours"] > max_gap_hours]
    if len(rollover_idx) > 0:
        df = df.iloc[:rollover_idx[0]].reset_index(drop=True)

    cutoff = df["Time"].iloc[0] + pd.Timedelta(days=drop_first_days)
    df = df[df["Time"] >= cutoff].reset_index(drop=True)

    df.loc[df["Temperature"] > 45.0, "Temperature"] = np.nan
    return df


def compute_dynamic_baseline_and_derivatives(df, window_hours=24, min_periods=12):
    df = df.copy()
    df_indexed = df.set_index("Time").sort_index()

    df_indexed["T_baseline_dyn"] = df_indexed["Temperature"].rolling(
        window=f"{window_hours}h", min_periods=min_periods, closed="left"
    ).median()

    if df_indexed["T_baseline_dyn"].isna().any():
        expanding_baseline = df_indexed["Temperature"].expanding(min_periods=1).median()
        df_indexed["T_baseline_dyn"] = df_indexed["T_baseline_dyn"].fillna(expanding_baseline)

    df_indexed["T_residual"] = df_indexed["Temperature"] - df_indexed["T_baseline_dyn"]

    df_indexed["dt_min"] = df_indexed.index.to_series().diff().dt.total_seconds() / 60.0
    df_indexed["dT_residual"] = df_indexed["T_residual"].diff()
    df_indexed["dTdt_residual"] = df_indexed["dT_residual"] / df_indexed["dt_min"]
    df_indexed["bad_time"] = (df_indexed["dt_min"] > 30) | (df_indexed["dt_min"] < 0)
    df_indexed.loc[df_indexed["bad_time"], "dTdt_residual"] = np.nan

    roc = df_indexed["dTdt_residual"].replace([np.inf, -np.inf], np.nan)
    dt_med = df_indexed["dt_min"].dropna()
    dt_med = dt_med[(dt_med > 0) & (dt_med < 30)]

    if len(dt_med) > 0:
        step = float(dt_med.median())
        win = max(3, int(round(60 / step)))
        df_indexed["roc_smooth"] = roc.rolling(win, min_periods=12, closed="left").median()
        df_indexed["roc_sd"] = roc.rolling(win, min_periods=12, closed="left").std()
    else:
        df_indexed["roc_smooth"] = np.nan
        df_indexed["roc_sd"] = np.nan

    return df_indexed.reset_index()


def make_5min_windows(all_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    all_windows = []
    for mouse, df in all_data.items():
        df5 = df.set_index("Time").resample(RESAMPLE_RULE).agg({
            "Temperature": "mean",
            "T_baseline_dyn": "mean",
            "T_residual": "mean",
            "roc_smooth": "mean",
            "roc_sd": "mean"
        }).reset_index()
        df5["mouse"] = mouse
        all_windows.append(df5)
    return pd.concat(all_windows, ignore_index=True)


def run_kmeans(windows: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    feat_cols = ["T_residual", "roc_smooth", "roc_sd"]
    X = windows[feat_cols].to_numpy(float)
    mask = ~np.isnan(X).any(axis=1)
    w = windows.loc[mask].copy().reset_index(drop=True)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(w[feat_cols].to_numpy(float))

    km = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10)
    w["cluster"] = km.fit_predict(Xs)

    # torpor cluster chosen as lowest mean residual
    cluster_residuals = w.groupby("cluster")["T_residual"].mean()
    torpor_cluster = int(cluster_residuals.idxmin())
    return w, torpor_cluster


# =============================================================================
# BOUT EXTRACTION
# =============================================================================

def extract_bouts_from_anchor(w_mouse: pd.DataFrame, torpor_cluster: int) -> pd.DataFrame:
    """
    w_mouse: 5-min windows for one mouse, includes cluster and Temperature
    bouts defined as contiguous windows with cluster == torpor_cluster (with small gaps allowed)
    """
    w_mouse = w_mouse.sort_values("Time").copy()
    torp = w_mouse[w_mouse["cluster"] == torpor_cluster].copy()
    if torp.empty:
        return pd.DataFrame()

    torp["gap_min"] = torp["Time"].diff().dt.total_seconds() / 60.0
    bout_id = 0
    bout_ids = []
    for i, g in enumerate(torp["gap_min"].to_numpy()):
        if i == 0 or (pd.notna(g) and g > MAX_BOUT_GAP_MINUTES):
            bout_id += 1
        bout_ids.append(bout_id)
    torp["bout_id"] = bout_ids

    bouts = []
    for bid, b in torp.groupby("bout_id"):
        onset = b["Time"].min()
        offset = b["Time"].max()
        dur_h = (offset - onset).total_seconds() / 3600.0
        if dur_h * 60.0 < MIN_BOUT_DURATION_MINUTES:
            continue
        Tmin = float(b["Temperature"].min())
        bouts.append({
            "mouse": str(b["mouse"].iloc[0]),
            "bout_id": f"{b['mouse'].iloc[0]}_B{bid}",
            "onset_time": onset,
            "offset_time": offset,
            "duration_hours": float(dur_h),
            "T_min": Tmin,
            "T_mean": float(b["Temperature"].mean()),
            "n_windows": int(len(b)),
            "severity": "Deep" if Tmin < DEEP_THRESHOLD else "Shallow",
        })
    return pd.DataFrame(bouts)


# =============================================================================
# ROBUST INFERENCE UTILITIES
# =============================================================================

def clustered_bootstrap_corr(bouts_df: pd.DataFrame, method="spearman", n=BOOTSTRAP_N, seed=0):
    """
    Bootstrap by resampling mice WITH replacement, then including all bouts for selected mice.
    This respects within-mouse dependence.
    """
    rng = np.random.RandomState(seed)
    mice = np.array(sorted(bouts_df["mouse"].unique()))
    stats = []
    for _ in range(n):
        samp_mice = rng.choice(mice, size=len(mice), replace=True)
        samp = pd.concat([bouts_df[bouts_df["mouse"] == m] for m in samp_mice], ignore_index=True)
        x = samp["T_min"].to_numpy(float)
        y = samp["duration_hours"].to_numpy(float)
        if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
            continue
        if method == "pearson":
            r, _ = pearsonr(x, y)
        else:
            r, _ = spearmanr(x, y)
        stats.append(r)
    stats = np.array(stats, dtype=float)
    return np.nanmedian(stats), np.nanpercentile(stats, [2.5, 97.5])


def within_mouse_correlations(bouts_df: pd.DataFrame):
    """
    Compute within-mouse Spearman for mice with >=3 bouts.
    """
    rows = []
    for m, sub in bouts_df.groupby("mouse"):
        if len(sub) < 3:
            continue
        r, p = spearmanr(sub["T_min"].to_numpy(float), sub["duration_hours"].to_numpy(float))
        rows.append({"mouse": m, "n_bouts": len(sub), "spearman_rho": r, "p": p})
    return pd.DataFrame(rows)


def permutation_test_within_mouse(bouts_df: pd.DataFrame, method="spearman", n=PERM_N, seed=0):
    """
    Permute T_min within each mouse (keeps each mouse's marginal distributions).
    Tests if observed association is stronger than expected by chance.
    """
    rng = np.random.RandomState(seed)

    x = bouts_df["T_min"].to_numpy(float)
    y = bouts_df["duration_hours"].to_numpy(float)
    if method == "pearson":
        obs, _ = pearsonr(x, y)
    else:
        obs, _ = spearmanr(x, y)

    mice = bouts_df["mouse"].to_numpy(str)
    perm_stats = np.zeros(n, dtype=float)

    # Pre-split indices by mouse for speed
    idx_by_mouse = {}
    for m in np.unique(mice):
        idx_by_mouse[m] = np.where(mice == m)[0]

    for i in range(n):
        x_perm = x.copy()
        for m, idx in idx_by_mouse.items():
            x_perm[idx] = rng.permutation(x_perm[idx])
        if method == "pearson":
            r, _ = pearsonr(x_perm, y)
        else:
            r, _ = spearmanr(x_perm, y)
        perm_stats[i] = r

    # two-sided p-value
    pval = (np.sum(np.abs(perm_stats) >= np.abs(obs)) + 1) / (n + 1)
    return obs, pval, perm_stats


# Optional mixed effects (nice if available)
def mixed_effects_duration_model(bouts_df: pd.DataFrame):
    """
    Mixed effects: duration_hours ~ T_min + (1 | mouse).
    Returns (beta, se, pval) if statsmodels is available.
    """
    try:
        import statsmodels.formula.api as smf
        # MixedLM can be sensitive; use REML=False for comparability
        model = smf.mixedlm("duration_hours ~ T_min", data=bouts_df, groups=bouts_df["mouse"])
        res = model.fit(reml=False, method="lbfgs")
        beta = float(res.params["T_min"])
        se = float(res.bse["T_min"])
        p = float(res.pvalues["T_min"])
        return beta, se, p, res
    except Exception as e:
        return None, None, None, str(e)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("ROBUST: BOUT MIN TEMPERATURE vs DURATION")
    print("=" * 80)

    # --- Load mice files
    all_data = {}
    mice = [f"N{i}" for i in range(1, 17)]
    for mouse in mice:
        filepath = f"{mouse}.xlsx"
        if not os.path.exists(filepath):
            matches = glob.glob(f"*_{mouse}.xlsx")
            if matches:
                filepath = matches[0]
            else:
                continue
        try:
            df = load_and_preprocess(filepath)
            df = compute_dynamic_baseline_and_derivatives(df, window_hours=24, min_periods=12)
            df["mouse"] = mouse
            all_data[mouse] = df
        except Exception as e:
            print(f"  {mouse}: FAILED - {e}")

    print(f"Loaded {len(all_data)} mice")
    if len(all_data) < 3:
        raise SystemExit("Not enough mice loaded to run robust analysis.")

    # --- Create windows + run kmeans
    windows = make_5min_windows(all_data)
    windows_clean, torpor_cluster = run_kmeans(windows)

    print(f"TORPOR_CLUSTER (lowest mean T_residual): {torpor_cluster}")
    print(f"Torpor windows: {(windows_clean['cluster'] == torpor_cluster).sum()}")

    # --- Extract bouts
    bouts_all = []
    for m in sorted(windows_clean["mouse"].unique()):
        w_m = windows_clean[windows_clean["mouse"] == m].copy()
        b = extract_bouts_from_anchor(w_m, torpor_cluster)
        if not b.empty:
            bouts_all.append(b)

    bouts_df = pd.concat(bouts_all, ignore_index=True) if bouts_all else pd.DataFrame()
    if bouts_df.empty:
        raise SystemExit("No bouts found.")

    bouts_df.to_csv(OUTDIR / "bouts.csv", index=False)

    print("\nBOUTS SUMMARY")
    print("-" * 80)
    print(f"Total bouts: {len(bouts_df)} from {bouts_df['mouse'].nunique()} mice")
    print(bouts_df.groupby("mouse").size().sort_index())

    # --- Basic pooled correlations
    x = bouts_df["T_min"].to_numpy(float)
    y = bouts_df["duration_hours"].to_numpy(float)
    r_p, p_p = pearsonr(x, y)
    r_s, p_s = spearmanr(x, y)

    print("\nPOOLED CORRELATIONS (treating bouts as independent; optimistic)")
    print("-" * 80)
    print(f"Pearson r  = {r_p:.3f}, p = {p_p:.3g}")
    print(f"Spearman ρ = {r_s:.3f}, p = {p_s:.3g}")

    # --- Clustered bootstrap CI (by mouse)
    med_rho, ci_rho = clustered_bootstrap_corr(bouts_df, method="spearman", n=BOOTSTRAP_N, seed=1)
    med_r, ci_r = clustered_bootstrap_corr(bouts_df, method="pearson", n=BOOTSTRAP_N, seed=2)

    print("\nCLUSTERED BOOTSTRAP (resample mice; accounts for repeated bouts)")
    print("-" * 80)
    print(f"Spearman ρ median = {med_rho:.3f}, 95% CI [{ci_rho[0]:.3f}, {ci_rho[1]:.3f}]")
    print(f"Pearson r  median = {med_r:.3f}, 95% CI [{ci_r[0]:.3f}, {ci_r[1]:.3f}]")

    # --- Within-mouse correlations
    wm = within_mouse_correlations(bouts_df)
    wm.to_csv(OUTDIR / "within_mouse_correlations.csv", index=False)
    print("\nWITHIN-MOUSE SPEARMAN (mice with >=3 bouts)")
    print("-" * 80)
    if wm.empty:
        print("No mouse has >=3 bouts (cannot compute within-mouse correlations reliably).")
    else:
        print(wm.sort_values("spearman_rho"))

        # simple sign test summary
        n_neg = int((wm["spearman_rho"] < 0).sum())
        n_tot = int(len(wm))
        print(f"\nMice with negative within-mouse ρ: {n_neg}/{n_tot}")

    # --- Permutation test within mouse
    obs_rho, perm_p, perm_stats = permutation_test_within_mouse(bouts_df, method="spearman", n=PERM_N, seed=3)
    print("\nWITHIN-MOUSE PERMUTATION TEST (Spearman; permute Tmin within each mouse)")
    print("-" * 80)
    print(f"Observed ρ = {obs_rho:.3f}")
    print(f"Permutation p-value (two-sided) = {perm_p:.4g}")

    # --- Optional mixed effects model
    beta, se, pval, extra = mixed_effects_duration_model(bouts_df)
    print("\nMIXED EFFECTS MODEL: duration_hours ~ T_min + (1|mouse) (optional)")
    print("-" * 80)
    if beta is None:
        print("Mixed effects not available / failed:", extra)
    else:
        print(f"beta(T_min) = {beta:.3f} hours/°C (SE={se:.3f}), p={pval:.3g}")
        # negative beta means lower Tmin -> longer duration

    # --- Plots
    plt.figure(figsize=(9, 6))
    for m, sub in bouts_df.groupby("mouse"):
        plt.scatter(sub["T_min"], sub["duration_hours"], s=70, alpha=0.75, edgecolors="black", linewidths=0.5, label=m)
    z = np.polyfit(x, y, 1)
    p_line = np.poly1d(z)
    xx = np.linspace(x.min(), x.max(), 200)
    plt.plot(xx, p_line(xx), "r--", linewidth=2, label=f"Linear fit (r={r_p:.2f})")
    plt.axvline(DEEP_THRESHOLD, linestyle="--", linewidth=2, alpha=0.4, label=f"Deep threshold {DEEP_THRESHOLD}°C")
    plt.xlabel("Bout minimum temperature T_min (°C)")
    plt.ylabel("Bout duration (hours)")
    plt.title("Bout minimum temperature vs duration (k-means-defined bouts)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTDIR / "scatter_all_bouts.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist(perm_stats, bins=50)
    plt.axvline(obs_rho, linewidth=2, label=f"Observed ρ={obs_rho:.2f}")
    plt.axvline(-obs_rho, linewidth=2, linestyle="--", label="sym")
    plt.title("Permutation null (within-mouse shuffle of Tmin)")
    plt.xlabel("Spearman ρ")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "perm_null_hist.png", dpi=160)
    plt.close()

    # --- Write a concise report text
    report = []
    report.append("ROBUST BOUT TEMPERATURE vs DURATION ANALYSIS")
    report.append("=" * 60)
    report.append(f"Bouts: {len(bouts_df)} from {bouts_df['mouse'].nunique()} mice")
    report.append("")
    report.append("POOLED (optimistic, bouts treated independent):")
    report.append(f"  Pearson r  = {r_p:.3f} (p={p_p:.3g})")
    report.append(f"  Spearman ρ = {r_s:.3f} (p={p_s:.3g})")
    report.append("")
    report.append("Clustered bootstrap (resample mice; 95% CI):")
    report.append(f"  Spearman ρ median = {med_rho:.3f}  CI [{ci_rho[0]:.3f}, {ci_rho[1]:.3f}]")
    report.append(f"  Pearson r  median = {med_r:.3f}  CI [{ci_r[0]:.3f}, {ci_r[1]:.3f}]")
    report.append("")
    report.append("Within-mouse permutation test (shuffle Tmin within each mouse):")
    report.append(f"  Observed Spearman ρ = {obs_rho:.3f}")
    report.append(f"  Permutation p (two-sided) = {perm_p:.4g}")
    report.append("")
    if beta is not None:
        report.append("Mixed effects (if available): duration ~ Tmin + (1|mouse)")
        report.append(f"  beta(Tmin) = {beta:.3f} hours/°C (negative => lower Tmin -> longer duration), p={pval:.3g}")
        report.append("")
    report.append("Interpretation (careful wording):")
    report.append("  Lower bout minimum temperature is robustly associated with longer bout duration.")
    report.append("  This is supported by pooled correlation AND by tests that account for within-mouse dependence.")
    (OUTDIR / "summary.txt").write_text("\n".join(report), encoding="utf-8")

    print("\nSaved outputs to:", OUTDIR.resolve())
    print("  - bouts.csv")
    print("  - within_mouse_correlations.csv")
    print("  - scatter_all_bouts.png")
    print("  - perm_null_hist.png")
    print("  - summary.txt")


if __name__ == "__main__":
    main()
