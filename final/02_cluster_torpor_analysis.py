# 02_cluster_torpor_analysis.py
# Read clustering outputs (GMM CSV + KMeans XLSX), rank clusters, choose torpor cluster,
# and export torpor flags + diagnostics.
#
# Expected inputs:
#   GMM dir:    clusters_all_samples.csv  (must contain label_gmm + either T_rel or T_residual)
#   KMeans dir: fixed_dynamic_clusters.xlsx (must contain cluster or label_kmeans + either T_residual or T_rel)
#
# Outputs:
#   out_dir/torpor_selection.json
#   out_dir/GMM_ranking.csv
#   out_dir/KMEANS_ranking.csv
#   out_dir/GMM__clusters_with_torpor_flag.csv
#   out_dir/KMEANS__clusters_with_torpor_flag.csv
#   out_dir/ALL__clusters_with_torpor_flag.csv (stacked)

import os
import json
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# Config
# -------------------------
@dataclass
class Config:
    gmm_dir: str = r"C:\Users\mayue\Desktop\Wavelet\out_cluster\gmm"
    kmeans_dir: str = r"C:\Users\mayue\Desktop\Wavelet\out_cluster\kmeans"
    out_dir: str = r"C:\Users\mayue\Desktop\Wavelet\out_cluster\analysis"

    # files (fixed names to match your folders)
    gmm_file: str = "clusters_all_samples.csv"
    kmeans_file: str = "fixed_dynamic_clusters.xlsx"

    # columns
    gmm_label_col: str = "label_gmm"
    kmeans_label_col_candidates: Tuple[str, ...] = ("label_kmeans", "cluster", "label", "kmeans")

    # torpor ranking: combine "low tail" + "median drop"
    # score = w_tail * (-p10) + w_med * (-median)
    w_tail: float = 0.6
    w_med: float = 0.4

    # plotting
    do_plots: bool = True
    plot_max_points: int = 120000  # cap for scatter/plots
    fig_dpi: int = 170


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def read_table(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path}")


def pick_first_existing_col(df: pd.DataFrame, candidates: Tuple[str, ...]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def normalize_columns_for_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unify residual column name to `T_rel` for analysis.
    Accepts: T_rel or T_residual.
    """
    df = df.copy()

    # time column normalization (optional)
    if "Time" in df.columns and "timestamp" not in df.columns:
        df = df.rename(columns={"Time": "timestamp"})

    # unify T_rel
    if "T_rel" not in df.columns:
        if "T_residual" in df.columns:
            df["T_rel"] = pd.to_numeric(df["T_residual"], errors="coerce")
        else:
            # last-resort: try temperature minus baseline if both exist
            if ("Temperature" in df.columns) and ("T_baseline_dyn" in df.columns):
                t = pd.to_numeric(df["Temperature"], errors="coerce")
                b = pd.to_numeric(df["T_baseline_dyn"], errors="coerce")
                df["T_rel"] = t - b

    # numeric coercion
    if "T_rel" in df.columns:
        df["T_rel"] = pd.to_numeric(df["T_rel"], errors="coerce")

    return df


def cap_df(df: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    if len(df) <= n:
        return df
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=n, replace=False)
    return df.iloc[idx].copy()


def rank_clusters(df: pd.DataFrame, label_col: str, cfg: Config, tag: str) -> Tuple[int, pd.DataFrame]:
    """
    Rank clusters by how "torpor-like" they are using T_rel distribution.
    Torpor should have very negative residual (low median and low tail).
    """
    if label_col not in df.columns:
        raise ValueError(f"{tag} clusters file must contain '{label_col}' column.")

    if "T_rel" not in df.columns:
        raise ValueError(f"{tag} clusters file must contain 'T_rel' (or 'T_residual' to map).")

    d = df[[label_col, "T_rel"]].dropna().copy()
    d[label_col] = pd.to_numeric(d[label_col], errors="coerce")
    d = d.dropna(subset=[label_col])
    d[label_col] = d[label_col].astype(int)

    rows = []
    for k in sorted(d[label_col].unique()):
        sub = d[d[label_col] == k]["T_rel"].to_numpy(dtype=float)
        if len(sub) < 20:
            continue
        p10 = float(np.nanpercentile(sub, 10))
        med = float(np.nanmedian(sub))
        n = int(len(sub))

        score = cfg.w_tail * (-p10) + cfg.w_med * (-med)
        rows.append(
            dict(
                cluster=int(k),
                n=n,
                pct=100.0 * n / max(1, len(d)),
                T_rel_p10=p10,
                T_rel_median=med,
                score=score,
            )
        )

    rank_df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    if rank_df.empty:
        raise RuntimeError(f"{tag}: no valid clusters to rank (maybe too many NaNs).")

    torpor_cluster = int(rank_df.iloc[0]["cluster"])
    return torpor_cluster, rank_df


def add_torpor_flag(df: pd.DataFrame, label_col: str, torpor_cluster: int, method_name: str) -> pd.DataFrame:
    out = df.copy()
    out["method"] = method_name
    out["torpor_cluster"] = int(torpor_cluster)
    out["is_torpor"] = (pd.to_numeric(out[label_col], errors="coerce") == torpor_cluster).astype(int)
    return out


def plot_box(df: pd.DataFrame, label_col: str, out_path: str, title: str):
    d = df[[label_col, "T_rel"]].dropna().copy()
    d[label_col] = pd.to_numeric(d[label_col], errors="coerce")
    d = d.dropna(subset=[label_col])
    d[label_col] = d[label_col].astype(int)

    labs = sorted(d[label_col].unique())
    data = [d[d[label_col] == k]["T_rel"].to_numpy(dtype=float) for k in labs]

    fig = plt.figure(figsize=(10.5, 4.2))
    # FIX: Matplotlib >=3.9 uses tick_labels instead of labels
    plt.boxplot(data, tick_labels=[f"c{k}" for k in labs], showfliers=False)
    plt.ylabel("T_rel (or mapped residual)")
    plt.title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def main():
    cfg = Config()
    ensure_dir(cfg.out_dir)

    # -------------------------
    # Load GMM
    # -------------------------
    print(f"\n[LOAD] GMM from: {cfg.gmm_dir}")
    gmm_path = os.path.join(cfg.gmm_dir, cfg.gmm_file)
    df_gmm = read_table(gmm_path)
    df_gmm = normalize_columns_for_analysis(df_gmm)
    print(f"[OK] Loaded: {os.path.basename(gmm_path)}  rows={len(df_gmm):,}  cols={len(df_gmm.columns)}")

    # -------------------------
    # Load KMeans
    # -------------------------
    print(f"\n[LOAD] KMeans from: {cfg.kmeans_dir}")
    km_path = os.path.join(cfg.kmeans_dir, cfg.kmeans_file)
    df_km = read_table(km_path)
    df_km = normalize_columns_for_analysis(df_km)

    km_label_col = pick_first_existing_col(df_km, cfg.kmeans_label_col_candidates)
    if km_label_col is None:
        raise ValueError(f"KMeans file must contain one of {cfg.kmeans_label_col_candidates}, got cols={list(df_km.columns)[:40]}")
    # Normalize label name to label_kmeans
    if km_label_col != "label_kmeans":
        df_km = df_km.rename(columns={km_label_col: "label_kmeans"})
    print(f"[OK] Loaded: {os.path.basename(km_path)}  rows={len(df_km):,}  cols={len(df_km.columns)}")

    # -------------------------
    # Rank & pick torpor cluster
    # -------------------------
    print(f"\n--- Analyzing GMM ({cfg.gmm_label_col}) ---")
    torpor_gmm, rank_gmm = rank_clusters(df_gmm, cfg.gmm_label_col, cfg, tag="GMM")
    print(f"[RESULT] Torpor cluster = {torpor_gmm}")
    print(f"         score={rank_gmm.iloc[0]['score']:.2f}, T_rel_p10={rank_gmm.iloc[0]['T_rel_p10']:.2f}, T_rel_med={rank_gmm.iloc[0]['T_rel_median']:.2f}")

    print(f"\n--- Analyzing KMEANS (label_kmeans) ---")
    torpor_km, rank_km = rank_clusters(df_km, "label_kmeans", cfg, tag="KMEANS")
    print(f"[RESULT] Torpor cluster = {torpor_km}")
    print(f"         score={rank_km.iloc[0]['score']:.2f}, T_rel_p10={rank_km.iloc[0]['T_rel_p10']:.2f}, T_rel_med={rank_km.iloc[0]['T_rel_median']:.2f}")

    # -------------------------
    # Save selection json + rankings
    # -------------------------
    sel = {"GMM": int(torpor_gmm), "KMEANS": int(torpor_km)}
    sel_path = os.path.join(cfg.out_dir, "torpor_selection.json")
    with open(sel_path, "w", encoding="utf-8") as f:
        json.dump(sel, f, indent=2)
    print(f"\n[DONE] torpor_selection.json saved to: {cfg.out_dir}")
    print(json.dumps(sel, indent=2))

    rank_gmm.to_csv(os.path.join(cfg.out_dir, "GMM_ranking.csv"), index=False)
    rank_km.to_csv(os.path.join(cfg.out_dir, "KMEANS_ranking.csv"), index=False)

    # -------------------------
    # Add torpor flags + save per-method
    # -------------------------
    out_gmm = add_torpor_flag(df_gmm, cfg.gmm_label_col, torpor_gmm, method_name="GMM")
    out_km = add_torpor_flag(df_km, "label_kmeans", torpor_km, method_name="KMEANS")

    out_gmm_path = os.path.join(cfg.out_dir, "GMM__clusters_with_torpor_flag.csv")
    out_km_path = os.path.join(cfg.out_dir, "KMEANS__clusters_with_torpor_flag.csv")
    out_gmm.to_csv(out_gmm_path, index=False)
    out_km.to_csv(out_km_path, index=False)

    # Combined (stack)
    all_path = os.path.join(cfg.out_dir, "ALL__clusters_with_torpor_flag.csv")
    pd.concat([out_gmm, out_km], ignore_index=True).to_csv(all_path, index=False)

    # -------------------------
    # Plots
    # -------------------------
    if cfg.do_plots:
        # cap to keep plots manageable
        pg = cap_df(out_gmm, cfg.plot_max_points, seed=0)
        pk = cap_df(out_km, cfg.plot_max_points, seed=1)

        plot_box(pg, cfg.gmm_label_col, os.path.join(cfg.out_dir, "box__GMM__T_rel.png"), "GMM: T_rel by cluster")
        plot_box(pk, "label_kmeans", os.path.join(cfg.out_dir, "box__KMEANS__T_rel.png"), "KMeans: T_rel by cluster")

    print(f"\n[SAVED] {out_gmm_path}")
    print(f"[SAVED] {out_km_path}")
    print(f"[SAVED] {all_path}")
    print(f"[SAVED] {os.path.join(cfg.out_dir, 'GMM_ranking.csv')}")
    print(f"[SAVED] {os.path.join(cfg.out_dir, 'KMEANS_ranking.csv')}")
    if cfg.do_plots:
        print(f"[SAVED] {os.path.join(cfg.out_dir, 'box__GMM__T_rel.png')}")
        print(f"[SAVED] {os.path.join(cfg.out_dir, 'box__KMEANS__T_rel.png')}")


if __name__ == "__main__":
    main()
