# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch, detrend

# =========================
# Path configurations (Ensure these match your local environment)
# =========================
DATA_DIR = Path(r"C:\Users\mayue\Desktop\Wavelet\Wavelet\Natural torpor N1-16")
SVM_CSV  = Path(r"C:\Users\mayue\Desktop\Wavelet\out_final\discovery_N16_final.csv")
OUT_ROOT = Path(r"C:\Users\mayue\Desktop\Wavelet\feature")
OUT_DIR  = OUT_ROOT / "Natural_torpor_N1-16_V2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Experiment timeline settings (Aligned with SVM)
# =========================
FAST_START = pd.Timestamp("2023-09-05 12:00:00")
POST_HOURS = 48
PRE_HOURS = 24  # [V2 Upgrade] Include 24 hours pre-fasting for visual baseline comparison

BASELINE_DAYS = 2
TOD_SMOOTH_MIN = 30 # Increased smoothing to counter daily fluctuation noise

# =========================
# Signal processing parameters (Welch PSD)
# =========================
# [V2 Upgrade] Capped at 60 mins to remove low-frequency 1/f noise ceiling
PSD_WIN_MIN = 120 
PSD_STEP_MIN = 1
PEAK_PERIOD_MIN_RANGE = (10, 60) 

# [V2 Upgrade] Shortened from 360 to 180 mins to aggressively flatten background slow waves
LOCAL_DETREND_MIN = 180 
MIN_WIN_SAMPLES = 32
MIN_NPERSEG = 32

# =========================
# Hypothesis testing parameters (Pre-dip "dip-and-recovery" phenomenon)
# =========================
LOOKBACK_HOURS = 12
MIN_PREDIP_MIN = 20
RECOVERY_RATIO = 0.6  # Requires a 60% rebound after dip to be considered a valid recovery
RECOVER_STABLE_MIN = 10
RECOVER_WITHIN_MIN = 120

# =========================
# I/O and utility functions
# =========================
def read_temp_xlsx(xlsx: Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx, sheet_name=0)
    cols = {str(c).strip().lower(): c for c in df.columns}
    time_col, temp_col = None, None
    for k, c in cols.items():
        if k in ("time", "timestamp", "time stamp", "datetime", "date_time"): time_col = c
        if "temp" in k: temp_col = c

    if time_col is None or temp_col is None:
        raise ValueError(f"Missing required columns in: {xlsx.name}")

    out = df[[time_col, temp_col]].rename(columns={time_col: "Time", temp_col: "Tb"})
    out["Time"] = pd.to_datetime(out["Time"], errors="coerce")
    out["Tb"] = pd.to_numeric(out["Tb"], errors="coerce")
    return out.dropna(subset=["Time"]).sort_values("Time")

def resample_1min(df: pd.DataFrame) -> pd.DataFrame:
    x = df.set_index("Time")["Tb"].resample("1min").mean()
    x = x.interpolate(limit=30, limit_direction="both")
    return x.to_frame().reset_index()

def minute_of_day(ts: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(ts)
    return (dt.dt.hour.to_numpy() * 60 + dt.dt.minute.to_numpy()).astype(int)

def rolling_median(x: np.ndarray, win: int) -> np.ndarray:
    return pd.Series(x).rolling(win, center=True, min_periods=max(10, win//3)).median().to_numpy()

def circular_smooth_1440(arr_1440: np.ndarray, w: int) -> np.ndarray:
    out = np.full_like(arr_1440, np.nan, dtype=float)
    for t in range(1440):
        idx = [(t + k) % 1440 for k in range(-w, w + 1)]
        out[t] = np.nanmedian(arr_1440[idx])
    return out

def build_baseline_profile(df_1m: pd.DataFrame, t0: pd.Timestamp, days: int) -> np.ndarray:
    start = t0 - pd.Timedelta(days=days)
    base = df_1m[(df_1m["Time"] >= start) & (df_1m["Time"] < t0)].copy()
    if len(base) < 1000: 
        return None
    base["tod"] = minute_of_day(base["Time"])
    med = base.groupby("tod")["Tb"].median()
    prof = np.full(1440, np.nan, dtype=float)
    prof[med.index.to_numpy()] = med.to_numpy()
    prof = circular_smooth_1440(prof, TOD_SMOOTH_MIN)
    return prof

# =========================
# Signal features: Sliding window Power Spectral Density (PSD)
# =========================
def sliding_psd_features(ts: pd.Series, x: np.ndarray, fs_per_min: float) -> pd.DataFrame:
    win = int(round(PSD_WIN_MIN * fs_per_min))
    step = int(round(PSD_STEP_MIN * fs_per_min))
    if win < MIN_WIN_SAMPLES or step < 1 or len(x) < win:
        return pd.DataFrame()

    pmin, pmax = PEAK_PERIOD_MIN_RANGE
    f_lo, f_hi = 1.0 / pmax, 1.0 / pmin 

    def bandpower(f, pxx, fmin, fmax):
        m = (f >= fmin) & (f < fmax)
        if m.sum() < 2: return np.nan
        return float(np.trapezoid(pxx[m], f[m]))

    rows = []
    for s in range(0, len(x) - win + 1, step):
        e = s + win
        seg_raw = x[s:e]


        if not np.all(np.isfinite(seg_raw)):
            continue

        seg = detrend(seg_raw, type="linear")
        nper = min(256, len(seg))
        if nper < MIN_NPERSEG: continue

        f, pxx = welch(seg, fs=fs_per_min, nperseg=nper, noverlap=nper//2, detrend=False)
        m = (f >= f_lo) & (f <= f_hi)
        peak_period = np.nan
        if m.sum() >= 3:
            k = np.argmax(pxx[m])
            peak_period = float(1.0 / (f[m][k] + 1e-12))

        bp_10_60 = bandpower(f, pxx, 1/60, 1/10)
        p = pxx / (pxx.sum() + 1e-12)
        sent = float(-np.sum(p * np.log(p + 1e-12)) / np.log(len(p)))

        rows.append({
            "t_mid": ts.iloc[s + win//2],
            "peak_period_min": peak_period,
            "bp_10_60": bp_10_60,
            "spectral_entropy": sent
        })
    return pd.DataFrame(rows)

# =========================
# Core hypothesis validation: Pre-dip probe
# =========================
def segments(mask: np.ndarray):
    out = []
    i, n = 0, len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i + 1
        while j < n and mask[j]: j += 1
        out.append((i, j))
        i = j
    return out

def hypothesis_predip_before_deep(ts: pd.Series, resid: np.ndarray, fs: float,
                                 deep_onset_time, d1: float, deep_level: float):
    if deep_onset_time is None or not np.isfinite(d1): return []

    lookback = deep_onset_time - pd.Timedelta(hours=LOOKBACK_HOURS)
    idx = (ts >= lookback) & (ts <= deep_onset_time)
    if idx.sum() < 200: return []

    tseg = ts[idx].reset_index(drop=True)
    rseg = pd.Series(resid[idx]).reset_index(drop=True)

    dip_mask = rseg.to_numpy() <= -d1
    dip_segs = segments(dip_mask)

    minlen = int(round(MIN_PREDIP_MIN * fs))
    events = []
    for s, e in dip_segs:
        if (e - s) < minlen: continue
        
        dip_min = float(np.min(rseg.to_numpy()[s:e]))
        
        # Exclude extrema that have already reached deep torpor levels
        if np.isfinite(deep_level) and dip_min <= (deep_level - 0.3): continue

        # Dynamic relative threshold: 60% rebound
        dip_depth = -dip_min
        recover_thr = dip_min + (dip_depth * RECOVERY_RATIO) 
        
        stable = max(1, int(round(RECOVER_STABLE_MIN * fs)))
        lim = min(len(rseg), e + int(round(RECOVER_WITHIN_MIN * fs)))
        
        rec = None
        for k in range(e, lim - stable + 1):
            if np.all(rseg.to_numpy()[k:k+stable] >= recover_thr):
                rec = k
                break
                
        if rec is None: continue

        events.append({
            "predip_start": tseg.iloc[s],
            "predip_end": tseg.iloc[e-1],
            "predip_min_resid_C": dip_min,
            "predip_duration_min": float((e - s) / fs),
            "recovery_time": tseg.iloc[rec],
            "recovery_delay_min": float((rec - e) / fs),
            "lead_to_deep_min": float((deep_onset_time - tseg.iloc[e-1]).total_seconds() / 60.0)
        })
    return events

# =========================
# SVM state overlay utilities
# =========================
def _state_intervals(t: pd.Series, state: pd.Series, target: str):
    st = state.astype(str).str.lower().to_numpy()
    tt = pd.to_datetime(t).to_numpy()
    mask = (st == target.lower())
    intervals = []
    i, n = 0, len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i + 1
        while j < n and mask[j]: j += 1
        intervals.append((pd.Timestamp(tt[i]), pd.Timestamp(tt[j-1])))
        i = j
    return intervals

def get_torpor_marks(svm_df: pd.DataFrame, mouse: str, t1: pd.Timestamp, t2: pd.Timestamp):
    sub = svm_df[svm_df["mouse_id"].astype(str) == str(mouse)].copy()
    sub = sub[(sub["datetime"] >= t1) & (sub["datetime"] <= t2)].sort_values("datetime")
    if sub.empty: return None, None, [], []

    st = sub["state"].astype(str).str.lower()
    torpor_start = sub.loc[st != "normal", "datetime"].min() if (st != "normal").any() else None
    deep_onset = sub.loc[st == "deep", "datetime"].min() if (st == "deep").any() else None

    gray_spans = _state_intervals(sub["datetime"], sub["state"], "Gray")
    deep_spans = _state_intervals(sub["datetime"], sub["state"], "Deep")
    return torpor_start, deep_onset, gray_spans, deep_spans

# =========================
# V2 Refactored clean visualization
# =========================
def plot_clean_analysis(psd, post, predip_events, torpor_start, deep_onset, gray_spans, deep_spans, mdir, mouse):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Subplot 1: Raw Tb and Baseline
    axes[0].plot(post["Time"], post["Tb"], label="Tb", color='black', lw=1)
    axes[0].plot(post["Time"], post["Tb_base"], label="Baseline", color='gray', linestyle='--')
    axes[0].set_ylabel("Tb (°C)", fontweight='bold')
    axes[0].legend(loc="upper right")
    
    # Subplot 2: Residuals (Setting the stage for Pre-dip)
    axes[1].plot(post["Time"], post["resid"], color='darkred', lw=1, label="Tb Residual")
    axes[1].axhline(0, color='black', lw=0.8, linestyle='--')
    axes[1].set_ylabel("Residual Tb (°C)", fontweight='bold')
    axes[1].legend(loc="upper right")

    # Subplot 3: Frequency changes (Validating Hypothesis 1)
    if not psd.empty:
        axes[2].plot(psd["t_mid"], psd["peak_period_min"], color='blue', lw=1.2, label="PSD Peak Period")
        axes[2].set_ylabel("Peak Period (min)", fontweight='bold')
        axes[2].legend(loc="upper right")
    axes[2].set_xlabel("Time (Including 24h Pre-Fasting)", fontweight='bold')

    # Add unified SVM state backgrounds and event lines to all subplots
    for ax in axes:
        # [V2 Upgrade] Bold line to mark the exact moment fasting begins
        ax.axvline(FAST_START, color='black', linestyle='-', lw=2.5, alpha=0.8, label="Fasting Starts" if ax==axes[0] else "")
        
        for a, b in gray_spans: ax.axvspan(a, b, color='lightgray', alpha=0.3, label='Gray Phase' if ax==axes[0] and a==gray_spans[0][0] else "")
        for a, b in deep_spans: ax.axvspan(a, b, color='lightblue', alpha=0.5, label='Deep Torpor' if ax==axes[0] and a==deep_spans[0][0] else "")
        if torpor_start: ax.axvline(torpor_start, color='orange', linestyle='--', lw=2, label="Torpor Start" if ax==axes[0] else "")
        if deep_onset: ax.axvline(deep_onset, color='red', linestyle='-.', lw=2, label="Deep Onset" if ax==axes[0] else "")
        
        # Annotate Pre-dip zones
        for ev in predip_events:
            ax.axvline(ev["predip_start"], color='green', linestyle=':', lw=1.5)
            ax.axvline(ev["recovery_time"], color='purple', linestyle=':', lw=1.5)

    plt.suptitle(f"{mouse} Torpor Transition Analysis (V2)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(mdir / "plots" / "comprehensive_analysis_V2.png", dpi=200)
    plt.close()

# =========================
# Main
# =========================
def main():
    svm = pd.read_csv(SVM_CSV)
    svm["datetime"] = pd.to_datetime(svm["datetime"], errors="coerce")
    svm = svm.dropna(subset=["datetime"])

    xlsx_files = sorted(DATA_DIR.glob("N*.xlsx"))
    all_psd, all_predip, all_summary = [], [], []

    for xlsx in xlsx_files:
        mouse = xlsx.stem
        mdir = OUT_DIR / mouse
        (mdir / "plots").mkdir(parents=True, exist_ok=True)

        df = read_temp_xlsx(xlsx)
        df = resample_1min(df)

        # Baseline calculation: Strictly take BASELINE_DAYS before fasting
        prof = build_baseline_profile(df, FAST_START, BASELINE_DAYS)
        if prof is None:
            print(f"[WARN] {mouse}: Insufficient baseline data for {BASELINE_DAYS} days pre-fasting.")
            continue

        # [V2 Upgrade] Analysis Window: 24h pre-fasting to 48h post-fasting
        t1 = FAST_START - pd.Timedelta(hours=PRE_HOURS)
        t2 = FAST_START + pd.Timedelta(hours=POST_HOURS)
        post = df[(df["Time"] >= t1) & (df["Time"] <= t2)].copy()
        if len(post) < 500: continue

        tod = minute_of_day(post["Time"])
        post["Tb_base"] = prof[tod]
        post["resid"] = post["Tb"] - post["Tb_base"]

        # Slow trend filtering using shorter V2 window (180 mins)
        win_local = int(round(LOCAL_DETREND_MIN * 1.0))
        post["resid_trend"] = rolling_median(post["resid"].to_numpy(), win_local)
        post["resid_osc"] = post["resid"].to_numpy() - post["resid_trend"].to_numpy()

        # PSD calculation
        psd = sliding_psd_features(post["Time"], post["resid_osc"].to_numpy(), fs_per_min=1.0)
        psd["mouse_id"] = mouse
        all_psd.append(psd)

        # SVM integration
        torpor_start, deep_onset, gray_spans, deep_spans = get_torpor_marks(svm, mouse, t1, t2)

        # Estimate residual values at deep torpor levels
        deep_level = np.nan
        if deep_onset is not None and len(deep_spans) > 0:
            deep_mask = np.zeros(len(post), dtype=bool)
            tpost = post["Time"].to_numpy()
            for a, b in deep_spans:
                deep_mask |= (tpost >= np.datetime64(a)) & (tpost <= np.datetime64(b))
            vals = post.loc[deep_mask, "resid"].to_numpy()
            if len(vals) > 10: deep_level = float(np.nanmedian(vals))

        # Dynamically determine baseline noise threshold d1
        base_df = df[(df["Time"] >= (FAST_START - pd.Timedelta(days=BASELINE_DAYS))) & (df["Time"] < FAST_START)].copy()
        base_tod = minute_of_day(base_df["Time"])
        base_resid = base_df["Tb"].to_numpy() - prof[base_tod]
        # Determine anomaly depth based on the specific mouse's characteristics
        d1 = float(max(0.3, -np.quantile(base_resid, 0.01))) 

        predip_events = hypothesis_predip_before_deep(
            post["Time"], post["resid"].to_numpy(), 1.0, deep_onset, d1=d1, deep_level=deep_level
        )

        # Summary statistics
        summary = {
            "mouse_id": mouse,
            "torpor_start": torpor_start,
            "deep_onset": deep_onset,
            "d1_used": d1,
            "deep_level_est": deep_level,
            "predip_found": int(len(predip_events) > 0),
        }

        # [V2 Upgrade] Calculates early vs pre-deep frequency differences. 
        # 'Early' is now the first 6 hours POST fasting.
        fasting_t1 = FAST_START 
        if deep_onset is not None and not psd.empty:
            pre6 = deep_onset - pd.Timedelta(hours=6)
            early = psd[(psd["t_mid"] >= fasting_t1) & (psd["t_mid"] < (fasting_t1 + pd.Timedelta(hours=6)))]
            late  = psd[(psd["t_mid"] >= pre6) & (psd["t_mid"] <= deep_onset)]
            summary["peak_period_early_mean"] = float(np.nanmean(early["peak_period_min"])) if len(early) else np.nan
            summary["peak_period_predeep_mean"] = float(np.nanmean(late["peak_period_min"])) if len(late) else np.nan
            summary["delta_peak_period"] = summary["peak_period_predeep_mean"] - summary["peak_period_early_mean"]
        else:
            summary.update({"peak_period_early_mean": np.nan, "peak_period_predeep_mean": np.nan, "delta_peak_period": np.nan})
        all_summary.append(summary)

        # Draw optimized charts
        plot_clean_analysis(psd, post, predip_events, torpor_start, deep_onset, gray_spans, deep_spans, mdir, mouse)

        for ev in predip_events:
            all_predip.append({"mouse_id": mouse, **ev})

        print(f"[OK] {mouse} processed.")

    # Global outputs
    if all_psd: pd.concat(all_psd, ignore_index=True).to_csv(OUT_DIR / "ALL_post48h_sliding_psd.csv", index=False)
    pd.DataFrame(all_summary).to_csv(OUT_DIR / "ALL_summary.csv", index=False)
    pd.DataFrame(all_predip).to_csv(OUT_DIR / "ALL_predip_events.csv", index=False)
    print(f"[Done] Outputs saved to {OUT_DIR}")

if __name__ == "__main__":
    main()