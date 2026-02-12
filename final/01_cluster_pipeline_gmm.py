import os, re, json, time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

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


TIME_CANDS = ["datetime","time","timestamp","ts","date","DateTime","Time","Timestamp"]
TEMP_CANDS = ["Tb","T","temp","temperature","Temperature","body_temp","body_temperature"]


@dataclass
class Config:
    data_root: str = r"C:\Users\mayue\Desktop\Wavelet\Wavelet\Natural torpor N1-16"
    out_dir: str = r"C:\Users\mayue\Desktop\Wavelet\out_cluster\gmm"

    dayfirst: bool = True
    resample_minutes: int = 5
    max_gap_hours: float = 24.0
    drop_first_days: int = 3

    baseline_window_hours: float = 24.0
    baseline_min_periods: int = 12
    baseline_closed_left: bool = True

    roc_window_minutes: int = 60
    roc_min_periods: int = 12
    roc_center: bool = True

    n_clusters: int = 4
    gmm_covariance_type: str = "full"
    gmm_reg_covar: float = 1e-6
    gmm_random_state: int = 42

    do_umap: bool = True
    umap_fit_n: int = 60000
    umap_n_neighbors: int = 50
    umap_min_dist: float = 0.10
    umap_random_state: int = 0

    do_global_plots: bool = True
    do_per_mouse_timeseries: bool = True
    max_points_per_mouse_plot: int = 120000
    per_mouse_window_start: str = "2023-09-03 00:00:00"


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def save_json(obj: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def _pick_col(cols: List[str], cands: List[str]) -> Optional[str]:
    m = {c.lower(): c for c in cols}
    for k in cands:
        if k.lower() in m:
            return m[k.lower()]
    for c in cols:
        cl = c.lower()
        if any(k.lower() in cl for k in cands):
            return c
    return None


def _infer_mouse_id(path: str) -> Optional[int]:
    base = os.path.basename(path)
    m = re.search(r"^N(\d{1,2})\.(xlsx|xls|csv)$", base, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def discover_files(root: str) -> List[str]:
    out = []
    for r, _, fs in os.walk(root):
        for fn in fs:
            if fn.lower().endswith((".xlsx", ".xls", ".csv")):
                out.append(os.path.join(r, fn))
    return sorted(out)


def parse_time(s: pd.Series, dayfirst: bool) -> pd.Series:
    ts = pd.to_datetime(s, errors="coerce", dayfirst=dayfirst)
    if ts.notna().mean() >= 0.7:
        return ts
    ts2 = pd.to_datetime(s, errors="coerce", dayfirst=not dayfirst)
    return ts2 if ts2.isna().mean() < ts.isna().mean() else ts


def read_best_sheet(path: str) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    best, best_score = None, -1
    for sh in xls.sheet_names:
        try:
            df = pd.read_excel(path, sheet_name=sh)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        tcol = _pick_col(df.columns.tolist(), TIME_CANDS)
        ycol = _pick_col(df.columns.tolist(), TEMP_CANDS)
        score = int(tcol is not None) + int(ycol is not None)
        if score > best_score:
            best_score, best = score, df
        if score == 2:
            return df
    if best is None:
        raise ValueError(f"no readable sheet: {path}")
    return best


def load_ts(path: str, cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(path) if path.lower().endswith(".csv") else read_best_sheet(path)
    tcol = _pick_col(df.columns.tolist(), TIME_CANDS)
    ycol = _pick_col(df.columns.tolist(), TEMP_CANDS)
    if tcol is None or ycol is None:
        raise ValueError(f"cannot infer Time/Temp columns in {os.path.basename(path)}")
    out = pd.DataFrame({
        "timestamp": parse_time(df[tcol], cfg.dayfirst),
        "T": pd.to_numeric(df[ycol], errors="coerce")
    })
    out = out.dropna(subset=["timestamp", "T"]).sort_values("timestamp").reset_index(drop=True)
    return out


def truncate_at_gap(df: pd.DataFrame, max_gap_hours: float) -> Tuple[pd.DataFrame, dict]:
    if df.empty:
        return df, {"truncated_at_gap": False}
    gap = df["timestamp"].diff().dt.total_seconds() / 3600.0
    idx = np.where(gap.to_numpy() > max_gap_hours)[0]
    if len(idx) == 0:
        return df, {"truncated_at_gap": False}
    cut = int(idx[0])
    return df.iloc[:cut].reset_index(drop=True), {"truncated_at_gap": True, "cut_idx": cut, "gap_hours": float(gap.iloc[cut])}


def drop_first_days(df: pd.DataFrame, days: int) -> Tuple[pd.DataFrame, dict]:
    if df.empty or days <= 0:
        return df, {"drop_first_days": 0}
    t0 = df["timestamp"].iloc[0]
    keep = t0 + pd.Timedelta(days=int(days))
    return df[df["timestamp"] >= keep].reset_index(drop=True), {"drop_first_days": int(days), "drop_cut_ts": keep}


def resample_5min(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    dfi = df.set_index("timestamp").sort_index()
    dfr = dfi.resample(f"{minutes}min").mean(numeric_only=True)
    dfr["T"] = dfr["T"].ffill(limit=1)
    return dfr.dropna(subset=["T"]).reset_index()


def dynamic_baseline(df_5: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    d = df_5.set_index("timestamp").sort_index()
    closed = "left" if cfg.baseline_closed_left else None
    base = d["T"].rolling(f"{cfg.baseline_window_hours}h", min_periods=cfg.baseline_min_periods, closed=closed).median()
    base = base.fillna(d["T"].expanding(min_periods=1).median())
    d["T_baseline_dyn"] = base
    d["T_residual"] = d["T"] - d["T_baseline_dyn"]
    return d.reset_index()


def roc_features(df_5: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    d = df_5.sort_values("timestamp").copy()
    dt = float(cfg.resample_minutes)
    d["roc_inst"] = d["T"].diff() / dt
    win = max(3, int(round(cfg.roc_window_minutes / dt)))
    d["roc_smooth"] = d["roc_inst"].rolling(win, min_periods=cfg.roc_min_periods, center=cfg.roc_center).median()
    d["roc_sd"] = d["roc_inst"].rolling(win, min_periods=cfg.roc_min_periods, center=cfg.roc_center).std()
    return d


def build_features(df_raw: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, dict]:
    d0, meta0 = truncate_at_gap(df_raw, cfg.max_gap_hours)
    d1, meta1 = drop_first_days(d0, cfg.drop_first_days)
    d2 = resample_5min(d1, cfg.resample_minutes)
    d3 = dynamic_baseline(d2, cfg)
    d4 = roc_features(d3, cfg)
    d4.loc[d4["T"] > 45.0, "T"] = np.nan
    d4 = d4.dropna(subset=["T", "T_baseline_dyn", "T_residual", "roc_smooth", "roc_sd"]).reset_index(drop=True)
    meta = {}
    meta.update(meta0); meta.update(meta1)
    return d4, meta


def umap_fit_transform(X: np.ndarray, cfg: Config) -> Optional[np.ndarray]:
    umap = try_import_umap()
    if umap is None:
        print("[WARN] umap-learn not installed; skip UMAP.")
        return None
    n = X.shape[0]
    fit_n = min(cfg.umap_fit_n, n)
    rng = np.random.default_rng(cfg.umap_random_state)
    X_fit = X[rng.choice(n, size=fit_n, replace=False)] if fit_n < n else X
    reducer = umap.UMAP(
        n_neighbors=cfg.umap_n_neighbors,
        min_dist=cfg.umap_min_dist,
        random_state=cfg.umap_random_state,
        verbose=False,
    )
    reducer.fit(X_fit)
    return reducer.transform(X)


def _cap_df_for_plot(d: pd.DataFrame, cap: int) -> pd.DataFrame:
    if len(d) <= cap:
        return d
    idx = np.linspace(0, len(d) - 1, cap).astype(int)
    return d.iloc[idx].copy()


def plot_mouse_timeseries_window(df_mouse: pd.DataFrame, label_col: str, out_path: str, cap: int, window_start: pd.Timestamp):
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
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.3))
    labs = sorted(df_all[label_col].dropna().unique().tolist())
    total = len(df_all)
    for lab in labs:
        sub = df_all[df_all[label_col] == lab]
        pct = 100.0 * len(sub) / max(1, total)
        axes[0].hist(sub["T_residual"], bins=140, alpha=0.35, label=f"c{lab} ({pct:.1f}%)")
        axes[1].hist(sub["roc_smooth"], bins=140, alpha=0.35, label=f"c{lab} ({pct:.1f}%)")
        axes[2].hist(sub["roc_sd"], bins=140, alpha=0.35, label=f"c{lab} ({pct:.1f}%)")
    axes[0].set_title("T_residual"); axes[1].set_title("roc_smooth"); axes[2].set_title("roc_sd")
    for ax in axes:
        ax.set_yscale("log")
        ax.legend(fontsize=9)
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

    files = discover_files(cfg.data_root)
    mouse_to_path: Dict[int, str] = {}
    for p in files:
        mid = _infer_mouse_id(p)
        if mid is not None:
            mouse_to_path.setdefault(mid, p)

    mice = sorted(mouse_to_path.keys())
    if not mice:
        raise RuntimeError("No N*.xlsx/csv found.")

    print(f"[OK] Found {len(mice)} mice: {mice}")

    all_rows, summary_rows = [], []
    for j, mid in enumerate(mice, start=1):
        path = mouse_to_path[mid]
        print(f"[LOAD] ({j}/{len(mice)}) mouse {mid}: {os.path.basename(path)}")
        df0 = load_ts(path, cfg)
        n_raw = len(df0)
        dfF, meta = build_features(df0, cfg)
        if dfF.empty:
            print(f"[WARN] mouse {mid} empty after preprocess")
            continue
        dfF["mouse_id"] = mid

        summary_rows.append({
            "mouse_id": mid,
            "path": path,
            "n_raw": n_raw,
            "n_features": len(dfF),
            **meta,
        })
        all_rows.append(dfF)

    if not all_rows:
        raise RuntimeError("All mice empty after preprocess. Check parsing/drop_first_days/gap.")

    df_all = pd.concat(all_rows, ignore_index=True)
    pd.DataFrame(summary_rows).to_csv(os.path.join(cfg.out_dir, "data_summary.csv"), index=False)
    print(f"[SAVED] {os.path.join(cfg.out_dir, 'data_summary.csv')}")

    feat_cols = ["T_residual", "roc_smooth", "roc_sd"]
    X = df_all[feat_cols].to_numpy(dtype=float)
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

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

    emb = umap_fit_transform(Xz, cfg) if cfg.do_umap else None
    if emb is not None:
        df_all["umap1"], df_all["umap2"] = emb[:, 0], emb[:, 1]

    df_all.to_csv(os.path.join(cfg.out_dir, "clusters_all_samples.csv"), index=False)
    print(f"[SAVED] {os.path.join(cfg.out_dir, 'clusters_all_samples.csv')}")

    save_json({
        "feature_cols": feat_cols,
        "dayfirst": cfg.dayfirst,
        "drop_first_days": cfg.drop_first_days,
        "max_gap_hours": cfg.max_gap_hours,
        "resample_minutes": cfg.resample_minutes,
        "baseline_window_hours": cfg.baseline_window_hours,
        "baseline_closed_left": cfg.baseline_closed_left,
        "roc_window_minutes": cfg.roc_window_minutes,
        "roc_center": cfg.roc_center,
        "n_clusters": cfg.n_clusters,
        "silhouette_gmm_on_Xz": sil,
        "per_mouse_window_start": cfg.per_mouse_window_start,
    }, os.path.join(cfg.out_dir, "config_cluster.json"))

    if cfg.do_global_plots:
        plot_distributions(df_all, "label_gmm", os.path.join(cfg.out_dir, "dist__label_gmm.png"))
        if emb is not None:
            plot_umap(emb, df_all["label_gmm"].to_numpy(), os.path.join(cfg.out_dir, "umap_scatter__label_gmm.png"), "UMAP colored by label_gmm")

    if cfg.do_per_mouse_timeseries:
        for mid in sorted(df_all["mouse_id"].unique()):
            d = df_all[df_all["mouse_id"] == mid].copy()
            out_gmm = os.path.join(viz_dir, f"mouse_N{int(mid):02d}__timeseries__gmm__from_0904.png")
            plot_mouse_timeseries_window(d, "label_gmm", out_gmm, cap=cfg.max_points_per_mouse_plot, window_start=window_start)

    print("\nCluster sizes (label_gmm):")
    for k in range(cfg.n_clusters):
        print(f"  cluster {k}: {(df_all['label_gmm']==k).sum()}")

    print(f"\n[DONE] Outputs in: {cfg.out_dir}")


if __name__ == "__main__":
    main()
