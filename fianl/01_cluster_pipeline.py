# 01_cluster_pipeline.py  (xlsx/csv supported, clustering + per-mouse timeseries viz; NO SVM)
import os
import re
import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# ---- stability (Windows OpenMP issues) ----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def try_import_umap():
    try:
        import umap
        return umap
    except Exception:
        return None


def try_import_hmmlearn():
    try:
        from hmmlearn.hmm import GaussianHMM
        return GaussianHMM
    except Exception:
        return None


@dataclass
class Config:
    data_root: str = r"C:\Users\mayue\Desktop\Wavelet\Wavelet\Natural torpor N1-16"
    out_dir: str = r"C:\Users\mayue\Desktop\Wavelet\out_cluster"

    # --- IMPORTANT: date parsing ---
    # If your Excel/csv uses dd/mm like 05/09/2023 meaning 5 Sep 2023, set dayfirst=True.
    dayfirst: bool = True

    # --- time handling ---
    resample_minutes: int = 5

    # --- HARD cold-start cut: drop first N hours no matter what (recommended 24~48h) ---
    drop_first_hours: float = 36.0

    # --- AUTO cold-start removal (runs AFTER hard cut, optional) ---
    do_auto_coldstart: bool = True
    cold_window_hours: float = 6.0
    cold_require_hours: float = 12.0
    cold_min_temp: float = 35.0      # stricter to detect stable euthermia
    cold_max_std: float = 0.40
    cold_max_hours_cap: float = 72.0

    # --- baseline ---
    # 05/09 is Sep 5 (dd/mm), so ISO should be 2023-09-05
    baseline_end_date: str = "2023-09-05"   # exclusive
    baseline_quantile: float = 0.70         # use top (1-q) fraction to avoid torpor contaminating baseline

    # --- features ---
    slope_window_minutes: int = 30          # rolling linear regression window in physical time
    delay_minutes: int = 15
    embed_dim: int = 3
    slope_clip: float = 5.0                # before sign-log1p

    # --- clustering ---
    n_clusters: int = 4
    gmm_covariance_type: str = "full"
    gmm_reg_covar: float = 1e-6
    gmm_random_state: int = 0

    # --- HMM optional ---
    do_hmm: bool = True
    hmm_n_iter: int = 100
    hmm_random_state: int = 0
    hmm_covariance_type: str = "full"

    # --- UMAP optional (viz only) ---
    do_umap: bool = True
    umap_fit_n: int = 60000
    umap_n_neighbors: int = 50
    umap_min_dist: float = 0.10
    umap_random_state: int = 0
    umap_verbose: bool = True

    # --- plots ---
    max_points_per_mouse_plot: int = 120000   # cap for plotting
    do_global_plots: bool = True              # dist__*.png and umap_scatter__*.png
    do_per_mouse_timeseries: bool = True


    per_mouse_window_start: str = "2023-09-03 00:00:00"


TIME_CANDIDATES = ["datetime", "time", "timestamp", "ts", "date", "DateTime", "Time", "Timestamp"]
TEMP_CANDIDATES = ["Tb", "T", "temp", "temperature", "Temperature", "body_temp", "body_temperature"]


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def save_json(obj: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _pick_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    cols_l = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in cols_l:
            return cols_l[cand.lower()]
    for c in cols:
        cl = c.lower()
        if any(k.lower() in cl for k in candidates):
            return c
    return None


def _infer_mouse_id_from_path(path: str) -> Optional[int]:
    base = os.path.basename(path)
    m = re.search(r"(?:^|[^\d])N(\d{1,2})(?:[^\d]|$)", base, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"(?:^|[^\d])(\d{1,2})(?:[^\d]|$)", base)
    if m:
        v = int(m.group(1))
        if 1 <= v <= 16:
            return v
    return None


def discover_data_files(data_root: str) -> List[str]:
    exts = (".csv", ".xlsx", ".xls")
    out = []
    for root, _, files in os.walk(data_root):
        for fn in files:
            if fn.lower().endswith(exts):
                out.append(os.path.join(root, fn))
    return sorted(out)


def _parse_time_series(raw: pd.Series, dayfirst: bool) -> pd.Series:
    # Key fix: dayfirst controls interpretation of strings like 05/09/2023
    ts = pd.to_datetime(raw, errors="coerce", utc=False, dayfirst=dayfirst)
    if ts.notna().sum() >= max(10, int(0.2 * len(raw))):
        return ts

    x = pd.to_numeric(raw, errors="coerce")
    if x.notna().sum() == 0:
        return ts

    # Excel serial dates fallback
    return pd.to_datetime(x, unit="D", origin="1899-12-30", errors="coerce")


def _read_excel_best_sheet(path: str) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    best_df, best_score = None, -1
    for sh in xls.sheet_names:
        try:
            df = pd.read_excel(path, sheet_name=sh)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        tcol = _pick_col(df.columns.tolist(), TIME_CANDIDATES)
        ycol = _pick_col(df.columns.tolist(), TEMP_CANDIDATES)
        score = int(tcol is not None) + int(ycol is not None)
        if score > best_score:
            best_score, best_df = score, df
        if score == 2:
            return df
    if best_df is None:
        raise ValueError(f"Excel file has no readable sheets: {path}")
    return best_df


def load_mouse_timeseries(path: str, cfg: Config) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = _read_excel_best_sheet(path)

    tcol = _pick_col(df.columns.tolist(), TIME_CANDIDATES)
    ycol = _pick_col(df.columns.tolist(), TEMP_CANDIDATES)
    if tcol is None or ycol is None:
        raise ValueError(f"Cannot infer time/temp columns in: {path}. cols={list(df.columns)[:30]}")

    ts = _parse_time_series(df[tcol], dayfirst=cfg.dayfirst)
    T = pd.to_numeric(df[ycol], errors="coerce")
    out = pd.DataFrame({"timestamp": ts, "T": T})
    out = out.dropna(subset=["timestamp", "T"]).sort_values("timestamp").reset_index(drop=True)
    return out


def resample_to_fixed_dt(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    dfi = df.set_index("timestamp").sort_index()
    dfr = dfi.resample(f"{minutes}min").mean(numeric_only=True)
    dfr["T"] = dfr["T"].ffill(limit=1)
    dfr = dfr.dropna(subset=["T"]).reset_index()
    return dfr


def hard_drop_first_hours(df_5min: pd.DataFrame, hours: float) -> Tuple[pd.DataFrame, dict]:
    if df_5min.empty or hours <= 0:
        return df_5min, {"hard_drop_hours": 0.0}
    t0 = df_5min["timestamp"].iloc[0]
    t_keep = t0 + pd.Timedelta(hours=float(hours))
    out = df_5min[df_5min["timestamp"] >= t_keep].reset_index(drop=True)
    dropped_hours = float((t_keep - t0).total_seconds() / 3600.0)
    return out, {"hard_drop_hours": dropped_hours}


def drop_cold_start_auto(df_5min: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, dict]:
    """
    Find first time where rolling window is 'stable euthermia' for a sustained period, then cut before it.
    Runs after hard_drop_first_hours.
    """
    if df_5min.empty:
        return df_5min, {"cold_start_dropped_hours": 0.0, "cold_start_reason": "empty"}

    dt_min = cfg.resample_minutes
    w = max(3, int(round((cfg.cold_window_hours * 60) / dt_min)))
    req = max(1, int(round((cfg.cold_require_hours * 60) / dt_min)))
    cap = max(1, int(round((cfg.cold_max_hours_cap * 60) / dt_min)))

    T = df_5min["T"].to_numpy(dtype=float)
    s = pd.Series(T)

    roll_min = s.rolling(w, min_periods=w).min()
    roll_std = s.rolling(w, min_periods=w).std()

    stable = (roll_min > cfg.cold_min_temp) & (roll_std < cfg.cold_max_std)
    stable = stable.fillna(False).to_numpy()

    run = np.zeros_like(stable, dtype=int)
    for i in range(len(stable)):
        run[i] = (run[i - 1] + 1) if (stable[i] and i > 0) else (1 if stable[i] else 0)

    idx_candidates = np.where(run >= req)[0]
    if len(idx_candidates) == 0:
        return df_5min, {"cold_start_dropped_hours": 0.0, "cold_start_reason": "no_stable_segment_found"}

    idx0 = int(idx_candidates[0])
    drop_idx = min(idx0, cap)

    t0 = df_5min["timestamp"].iloc[0]
    t_keep = df_5min["timestamp"].iloc[drop_idx]
    dropped_hours = float((t_keep - t0).total_seconds() / 3600.0)

    out = df_5min.iloc[drop_idx:].reset_index(drop=True)
    meta = {
        "cold_start_dropped_hours": dropped_hours,
        "cold_start_reason": "stable_window_detected",
        "cold_window_hours": cfg.cold_window_hours,
        "cold_require_hours": cfg.cold_require_hours,
        "cold_min_temp": cfg.cold_min_temp,
        "cold_max_std": cfg.cold_max_std,
    }
    return out, meta


def compute_baseline(df: pd.DataFrame, cfg: Config) -> float:
    """
    Baseline should represent euthermic reference, not torpor.
    Strategy:
      - use timestamps < baseline_end_date if possible
      - within that window, take temperatures above quantile(q) then median
      - fallback: first 24h then same quantile filter
    """
    # baseline_end_date is ISO, so parsing is unambiguous; still keep dayfirst for safety if user changes format
    end_ts = pd.to_datetime(cfg.baseline_end_date, dayfirst=cfg.dayfirst)
    sub = df[df["timestamp"] < end_ts]

    if len(sub) < 12:
        t0 = df["timestamp"].iloc[0]
        sub = df[df["timestamp"] < (t0 + pd.Timedelta(hours=24))]

    T = sub["T"].to_numpy(dtype=float)
    if len(T) < 12:
        return float(np.nanmedian(df["T"].to_numpy(dtype=float)))

    q = float(cfg.baseline_quantile)
    thr = np.nanquantile(T, q)
    T_hi = T[T >= thr]
    if len(T_hi) < 12:
        T_hi = T
    return float(np.nanmedian(T_hi))


def sign_log1p(x: np.ndarray, clip: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.clip(x, -clip, clip)
    return np.sign(x) * np.log1p(np.abs(x))


def rolling_regression_slope(y: np.ndarray, dt_min: float, window_min: int) -> np.ndarray:
    """
    Rolling linear regression slope over a fixed physical window.
    Output in °C/min.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    k = int(round(window_min / dt_min))
    if k < 3 or n < k:
        return np.full(n, np.nan)

    r = np.arange(k, dtype=float)
    t = r * dt_min
    sum_t = float(t.sum())
    sum_t2 = float((t * t).sum())
    denom = k * sum_t2 - sum_t * sum_t
    if denom <= 0:
        return np.full(n, np.nan)

    idx = np.arange(n, dtype=float)
    cy = np.concatenate([[0.0], np.cumsum(y)])
    ciy = np.concatenate([[0.0], np.cumsum(idx * y)])

    b = np.full(n, np.nan)
    for i in range(k - 1, n):
        s = i - k + 1
        sum_y = cy[i + 1] - cy[s]
        sum_iy = ciy[i + 1] - ciy[s]
        sum_ry = sum_iy - s * sum_y
        sum_ty = dt_min * sum_ry
        b[i] = (k * sum_ty - sum_t * sum_y) / denom
    return b


def build_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    dt_min = float(cfg.resample_minutes)
    baseline = compute_baseline(df, cfg)

    d = df.copy()
    d["T_base"] = baseline
    d["T_rel"] = d["T"] - baseline

    slope_raw = rolling_regression_slope(
        d["T_rel"].to_numpy(dtype=float),
        dt_min=dt_min,
        window_min=cfg.slope_window_minutes,
    )
    d["Slope_raw"] = slope_raw
    d["Slope"] = sign_log1p(slope_raw, clip=cfg.slope_clip)

    lag_steps = int(round(cfg.delay_minutes / dt_min))
    if lag_steps < 1 or cfg.embed_dim != 3:
        raise ValueError("Requires delay_minutes>=resample_minutes and embed_dim==3.")

    d["T_rel_lag15"] = d["T_rel"].shift(lag_steps)
    d["T_rel_lag30"] = d["T_rel"].shift(2 * lag_steps)

    d = d.dropna(subset=["T_rel", "Slope", "T_rel_lag15", "T_rel_lag30"]).reset_index(drop=True)
    return d


def umap_fit_transform(X: np.ndarray, cfg: Config) -> Optional[np.ndarray]:
    umap = try_import_umap()
    if umap is None:
        print("[WARN] umap-learn not installed; skip UMAP.")
        return None

    n = X.shape[0]
    fit_n = min(cfg.umap_fit_n, n)
    rng = np.random.default_rng(cfg.umap_random_state)
    X_fit = X[rng.choice(n, size=fit_n, replace=False)] if fit_n < n else X

    print(f"[UMAP] fit on {len(X_fit):,} samples ...")
    reducer = umap.UMAP(
        n_neighbors=cfg.umap_n_neighbors,
        min_dist=cfg.umap_min_dist,
        random_state=cfg.umap_random_state,
        verbose=cfg.umap_verbose,
    )
    t0 = time.time()
    reducer.fit(X_fit)
    print(f"[UMAP] fit done in {time.time()-t0:.1f}s")

    print(f"[UMAP] transform all {n:,} samples ...")
    t1 = time.time()
    emb = reducer.transform(X)
    print(f"[UMAP] transform done in {time.time()-t1:.1f}s")
    return emb


# -------------------------
# Plotting helpers
# -------------------------
def _cap_df_for_plot(d: pd.DataFrame, cap: int) -> pd.DataFrame:
    if len(d) <= cap:
        return d
    idx = np.linspace(0, len(d) - 1, cap).astype(int)
    return d.iloc[idx].copy()


def plot_mouse_timeseries_window(df_mouse: pd.DataFrame, label_col: str, out_path: str, cap: int, window_start: pd.Timestamp):
    """
    Single-mouse timeseries only.
    Plot from window_start (inclusive) to end of recording.
    """
    d = df_mouse.sort_values("timestamp").copy()
    d = d[d["timestamp"] >= window_start].copy()
    if d.empty:
        return

    d = _cap_df_for_plot(d, cap)

    t0 = window_start
    x = (d["timestamp"] - t0).dt.total_seconds() / 3600.0

    fig = plt.figure(figsize=(18, 4.5))
    plt.plot(x, d["T"], linewidth=1.1, alpha=0.35)
    plt.scatter(x, d["T"], s=10, c=d[label_col], alpha=0.90)

    mid = int(d["mouse_id"].iloc[0])
    plt.title(f"Mouse N{mid:02d}: T(t) with {label_col} | from {window_start} to end")
    plt.xlabel(f"Hours since {window_start.strftime('%Y-%m-%d %H:%M')}")
    plt.ylabel("Temperature (°C)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_distributions(df_all: pd.DataFrame, label_col: str, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3))
    labs = sorted(df_all[label_col].dropna().unique().tolist())
    total = len(df_all)

    for lab in labs:
        sub = df_all[df_all[label_col] == lab]
        pct = 100.0 * len(sub) / max(1, total)
        axes[0].hist(sub["T_rel"], bins=140, alpha=0.35, label=f"c{lab} ({pct:.1f}%)")
        axes[1].hist(sub["Slope"], bins=140, alpha=0.35, label=f"c{lab} ({pct:.1f}%)")

    axes[0].set_title("T_rel"); axes[1].set_title("Slope")
    for ax in axes:
        ax.set_yscale("log")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_umap(emb: np.ndarray, labels: np.ndarray, out_path: str, title: str):
    fig = plt.figure(figsize=(7.2, 6.6))
    plt.scatter(emb[:, 0], emb[:, 1], s=3.0, c=labels, alpha=0.75)
    plt.title(title)
    plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    cfg = Config()
    ensure_dir(cfg.out_dir)

    viz_dir = os.path.join(cfg.out_dir, "viz_per_mouse_from_0904")
    ensure_dir(viz_dir)

    window_start = pd.to_datetime(cfg.per_mouse_window_start, dayfirst=cfg.dayfirst)

    files = discover_data_files(cfg.data_root)
    if not files:
        raise RuntimeError(f"No data files (.xlsx/.xls/.csv) found under: {cfg.data_root}")

    mouse_to_path: Dict[int, str] = {}
    for p in files:
        mid = _infer_mouse_id_from_path(p)
        if mid is not None:
            mouse_to_path.setdefault(mid, p)

    mice = sorted(mouse_to_path.keys())
    if not mice:
        raise RuntimeError("Could not infer mouse IDs. Please name files like N1.xlsx ... N16.xlsx")

    print(f"[OK] Found {len(mice)} mice: {mice}")

    all_rows = []
    summary_rows = []

    for j, mid in enumerate(mice, start=1):
        path = mouse_to_path[mid]
        print(f"[LOAD] ({j}/{len(mice)}) mouse {mid}: {os.path.basename(path)}")
        df0 = load_mouse_timeseries(path, cfg)
        n_raw = len(df0)
        if df0.empty:
            print(f"[SKIP] Mouse {mid} empty after reading.")
            continue

        # 1) resample first to 5-min grid
        df2 = resample_to_fixed_dt(df0, cfg.resample_minutes)

        # 2) hard drop first N hours
        df2_hard, meta_hard = hard_drop_first_hours(df2, cfg.drop_first_hours)

        # 3) optional auto coldstart
        if cfg.do_auto_coldstart:
            df2_cut, meta_cold = drop_cold_start_auto(df2_hard, cfg)
        else:
            df2_cut, meta_cold = df2_hard, {"cold_start_dropped_hours": 0.0, "cold_start_reason": "disabled"}

        # 4) features built AFTER cuts
        dfF = build_features(df2_cut, cfg)
        if dfF.empty:
            print(f"[SKIP] Mouse {mid} empty after features.")
            continue

        dfF["mouse_id"] = mid

        summary_rows.append({
            "mouse_id": mid,
            "path": path,
            "n_raw": n_raw,
            "n_resampled": len(df2),
            "n_after_hard_drop": len(df2_hard),
            "n_after_coldstart": len(df2_cut),
            "n_features": len(dfF),
            **meta_hard,
            **meta_cold,
            "baseline_median_hiQ": float(dfF["T_base"].iloc[0]),
            "dayfirst": cfg.dayfirst,
        })
        all_rows.append(dfF)

    if not all_rows:
        raise RuntimeError("All mice became empty. Check timestamp parsing (dayfirst) and your input columns.")

    df_all = pd.concat(all_rows, ignore_index=True)
    df_summary = pd.DataFrame(summary_rows)

    df_summary.to_csv(os.path.join(cfg.out_dir, "data_summary.csv"), index=False)
    print(f"[SAVED] {os.path.join(cfg.out_dir, 'data_summary.csv')}")

    feat_cols = ["T_rel", "Slope", "T_rel_lag15", "T_rel_lag30"]
    X = df_all[feat_cols].to_numpy(dtype=float)

    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    # --- GMM ---
    gmm = GaussianMixture(
        n_components=cfg.n_clusters,
        covariance_type=cfg.gmm_covariance_type,
        reg_covar=cfg.gmm_reg_covar,
        random_state=cfg.gmm_random_state,
    )
    df_all["label_gmm"] = gmm.fit_predict(Xz)

    sil = None
    try:
        sil = float(silhouette_score(Xz, df_all["label_gmm"].to_numpy()))
    except Exception:
        pass

    # --- HMM (optional) ---
    if cfg.do_hmm:
        GaussianHMM = try_import_hmmlearn()
        if GaussianHMM is None:
            print("[WARN] hmmlearn not installed; skip HMM.")
        else:
            lengths = []
            Xz_seq = []
            for mid in sorted(df_all["mouse_id"].unique()):
                sub = df_all[df_all["mouse_id"] == mid]
                Xz_seq.append(scaler.transform(sub[feat_cols].to_numpy(dtype=float)))
                lengths.append(len(sub))
            Xz_cat = np.vstack(Xz_seq)

            print(f"[HMM] fitting on {Xz_cat.shape[0]:,} samples, lengths={len(lengths)} mice ...")
            hmm = GaussianHMM(
                n_components=cfg.n_clusters,
                covariance_type=cfg.hmm_covariance_type,
                n_iter=cfg.hmm_n_iter,
                random_state=cfg.hmm_random_state,
                verbose=False,
            )
            hmm.fit(Xz_cat, lengths=lengths)
            df_all["label_hmm"] = hmm.predict(Xz_cat, lengths=lengths)

    # --- UMAP (optional, viz only) ---
    emb = umap_fit_transform(Xz, cfg) if cfg.do_umap else None
    if emb is not None:
        df_all["umap1"], df_all["umap2"] = emb[:, 0], emb[:, 1]

    # --- Save main outputs ---
    df_all.to_csv(os.path.join(cfg.out_dir, "clusters_all_samples.csv"), index=False)
    print(f"[SAVED] {os.path.join(cfg.out_dir, 'clusters_all_samples.csv')}")

    save_json({
        "feature_cols": feat_cols,
        "resample_minutes": cfg.resample_minutes,
        "drop_first_hours": cfg.drop_first_hours,
        "do_auto_coldstart": cfg.do_auto_coldstart,
        "cold_start_rule": {
            "cold_window_hours": cfg.cold_window_hours,
            "cold_require_hours": cfg.cold_require_hours,
            "cold_min_temp": cfg.cold_min_temp,
            "cold_max_std": cfg.cold_max_std,
            "cold_max_hours_cap": cfg.cold_max_hours_cap,
        },
        "baseline_end_date": cfg.baseline_end_date,
        "baseline_quantile": cfg.baseline_quantile,
        "slope_window_minutes": cfg.slope_window_minutes,
        "delay_minutes": cfg.delay_minutes,
        "embed_dim": cfg.embed_dim,
        "n_clusters": cfg.n_clusters,
        "silhouette_gmm_on_Xz": sil,
        "per_mouse_window_start": cfg.per_mouse_window_start,
        "dayfirst": cfg.dayfirst,
    }, os.path.join(cfg.out_dir, "config_cluster.json"))
    print(f"[SAVED] {os.path.join(cfg.out_dir, 'config_cluster.json')}")

    # --- Global plots ---
    if cfg.do_global_plots:
        plot_distributions(df_all, "label_gmm", os.path.join(cfg.out_dir, "dist__label_gmm.png"))
        if "label_hmm" in df_all.columns:
            plot_distributions(df_all, "label_hmm", os.path.join(cfg.out_dir, "dist__label_hmm.png"))

        if emb is not None:
            plot_umap(
                emb,
                df_all["label_gmm"].to_numpy(),
                os.path.join(cfg.out_dir, "umap_scatter__label_gmm.png"),
                "UMAP colored by label_gmm",
            )
            if "label_hmm" in df_all.columns:
                plot_umap(
                    emb,
                    df_all["label_hmm"].to_numpy(),
                    os.path.join(cfg.out_dir, "umap_scatter__label_hmm.png"),
                    "UMAP colored by label_hmm",
                )

    # --- Per-mouse timeseries only (from 09/04 to end) ---
    if cfg.do_per_mouse_timeseries:
        uniq_mice = sorted(df_all["mouse_id"].unique().tolist())
        print(f"[VIZ] per-mouse timeseries -> {viz_dir} ({len(uniq_mice)} mice), window_start={window_start}")
        for mid in uniq_mice:
            d = df_all[df_all["mouse_id"] == mid].copy()

            out_gmm = os.path.join(viz_dir, f"mouse_N{int(mid):02d}__timeseries__gmm__from_0904.png")
            plot_mouse_timeseries_window(d, "label_gmm", out_gmm, cap=cfg.max_points_per_mouse_plot, window_start=window_start)

            if "label_hmm" in d.columns:
                out_hmm = os.path.join(viz_dir, f"mouse_N{int(mid):02d}__timeseries__hmm__from_0904.png")
                plot_mouse_timeseries_window(d, "label_hmm", out_hmm, cap=cfg.max_points_per_mouse_plot, window_start=window_start)

    # --- Print cluster sizes ---
    print("\nCluster sizes (label_gmm):")
    for k in range(cfg.n_clusters):
        print(f"  cluster {k}: {(df_all['label_gmm']==k).sum()}")

    if "label_hmm" in df_all.columns:
        print("\nCluster sizes (label_hmm):")
        for k in range(cfg.n_clusters):
            print(f"  cluster {k}: {(df_all['label_hmm']==k).sum()}")

    print(f"\n[DONE] Outputs in: {cfg.out_dir}")
    if cfg.do_per_mouse_timeseries:
        print(f"[DONE] Per-mouse timeseries in: {viz_dir}")


if __name__ == "__main__":
    main()
