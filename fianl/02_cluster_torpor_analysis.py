# 02_identify_torpor.py
# ------------------------------------------------------------------
# 功能：自动识别 Torpor 簇 (GMM & HMM 通用版)
# 核心升级：引入 P10 (10th Percentile) 评分，解决 HMM 混合状态识别问题。
# ------------------------------------------------------------------

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 尝试导入 seaborn 绘图库 (非必须)
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# -----------------------------
# 配置部分
# -----------------------------
DEFAULT_OUT_CLUSTER_DIR = Path(r"C:\Users\mayue\Desktop\Wavelet\out_cluster")
DEFAULT_CLUSTERS_CSV = DEFAULT_OUT_CLUSTER_DIR / "clusters_all_samples.csv"
DEFAULT_OUT_DIR = DEFAULT_OUT_CLUSTER_DIR / "out_torpor_identification"

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 核心算法函数
# -----------------------------
def get_contiguous_spans(mask: np.ndarray):
    """提取布尔序列中的连续片段 [(start, end), ...]"""
    padded = np.concatenate(([False], mask, [False]))
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    return list(zip(starts, ends))

def compute_bout_stats(df: pd.DataFrame, label_col: str, group_col: str = "mouse_id"):
    """计算每个簇的持续时间 (Bout Duration)"""
    rows = []
    # 确保数据按时间排序
    df = df.sort_values([group_col, "timestamp"]).reset_index(drop=True)

    # 1. 自动估算采样间隔 (dt)
    first_mouse = df[df[group_col] == df[group_col].iloc[0]]
    if len(first_mouse) > 1:
        dt_sec = pd.to_datetime(first_mouse["timestamp"]).diff().median().total_seconds()
        dt_min = dt_sec / 60.0 if (np.isfinite(dt_sec) and dt_sec > 0) else 5.0
    else:
        dt_min = 5.0
    
    print(f"[INFO] Detected sampling interval: {dt_min:.1f} min")

    # 2. 遍历计算 Bouts
    for mid, sub in df.groupby(group_col):
        lab = sub[label_col].values
        for k in np.unique(lab):
            mask = (lab == k)
            spans = get_contiguous_spans(mask)
            if not spans: continue
            
            # 计算时长
            lens = np.array([(end - start + 1) for start, end in spans], dtype=float)
            mins = lens * dt_min
            
            for m_val in mins:
                rows.append({"mouse_id": mid, "cluster": int(k), "bout_duration_min": float(m_val)})

    bout_df = pd.DataFrame(rows)
    if bout_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # 3. 汇总统计
    agg = bout_df.groupby("cluster").agg(
        n_bouts=("bout_duration_min", "count"),
        bout_min_median=("bout_duration_min", "median"),
        bout_min_p90=("bout_duration_min", lambda x: np.quantile(x, 0.90)), # 关键指标：P90时长
    ).reset_index()
    
    return bout_df, agg

def score_torpor_candidates_v2(cluster_stats: pd.DataFrame, bout_agg: pd.DataFrame):
    """
    V2 评分逻辑：同时兼容 GMM (纯净簇) 和 HMM (混合簇)
    """
    s = cluster_stats.copy()
    if bout_agg is not None and not bout_agg.empty:
        s = s.merge(bout_agg, on="cluster", how="left")
    else:
        s["bout_min_p90"] = 0.0
    s = s.fillna(0)

    # 鲁棒 Z-score 计算
    def robust_z(x):
        x = np.asarray(x, dtype=float)
        mu = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - mu))
        sd = mad * 1.4826
        return (x - mu) / (sd if sd > 1e-9 else 1.0)

    # 计算特征分数
    z_p10 = robust_z(s["T_rel_p10"])      # 最低温 (识别混合簇的关键)
    z_med = robust_z(s["T_rel_median"])   # 中位温
    z_dur = robust_z(s["bout_min_p90"])   # 持续时长

    # --- 评分公式 ---
    # 权重策略：
    # 1. P10 (-3.0): 只要包含深休眠样本，得分就暴涨 (针对 HMM)
    # 2. Median (-1.0): 整体最好也是低温 (针对 GMM)
    # 3. Duration (+1.0): 持续时间越长越好
    raw_score = (-3.0 * z_p10) + (-1.0 * z_med) + (1.0 * z_dur)

    # --- 安全检查 ---
    # 如果中位体温 > -0.5 (接近正常体温)，无论 P10 多低，都视为噪音或假阳性
    penalty_mask = s["T_rel_median"] > -0.5
    s["torpor_score"] = raw_score
    s.loc[penalty_mask, "torpor_score"] -= 1000.0

    return s.sort_values("torpor_score", ascending=False).reset_index(drop=True)

def plot_diagnostics(df: pd.DataFrame, bout_df: pd.DataFrame, label_col: str, out_dir: Path, prefix: str):
    """生成诊断图表：直方图、UMAP、箱线图"""
    ensure_dir(out_dir)
    clusters = sorted(df[label_col].dropna().unique())

    # 1. 直方图 (T_rel)
    plt.figure(figsize=(8, 4))
    for k in clusters:
        sub = df[df[label_col] == k]["T_rel"].dropna()
        if len(sub) > 5000: sub = sub.sample(5000, random_state=0)
        plt.hist(sub, bins=60, density=True, alpha=0.4, label=f"C{int(k)}")
    plt.title(f"{prefix}: T_rel Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_{label_col}_hist.png", dpi=150)
    plt.close()

    # 2. UMAP
    if "umap1" in df.columns:
        plt.figure(figsize=(6, 5))
        for k in clusters:
            sub = df[df[label_col] == k]
            if len(sub) > 5000: sub = sub.sample(5000, random_state=0)
            plt.scatter(sub["umap1"], sub["umap2"], s=2, alpha=0.6, label=f"C{int(k)}")
        plt.title(f"{prefix}: UMAP")
        plt.legend(markerscale=3)
        plt.tight_layout()
        plt.savefig(out_dir / f"{prefix}_{label_col}_umap.png", dpi=150)
        plt.close()

    # 3. Boxplot (时长)
    if not bout_df.empty:
        plt.figure(figsize=(8, 5))
        dat = bout_df[bout_df["bout_duration_min"] > 0]
        if HAS_SEABORN:
            sns.boxplot(data=dat, x="cluster", y="bout_duration_min", showfliers=False, palette="Set2")
        else:
            dat.boxplot(column="bout_duration_min", by="cluster", showfliers=False)
        plt.yscale("log")
        plt.title(f"{prefix}: Bout Duration (Log Scale)")
        plt.tight_layout()
        plt.savefig(out_dir / f"{prefix}_{label_col}_boxplot.png", dpi=150)
        plt.close()

# -----------------------------
# 主函数
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters_csv", type=str, default=str(DEFAULT_CLUSTERS_CSV))
    ap.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    args = ap.parse_args()

    clusters_csv = Path(args.clusters_csv)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    print(f"Loading: {clusters_csv}")
    if not clusters_csv.exists():
        print("[ERROR] File not found.")
        return

    df = pd.read_csv(clusters_csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "T_rel"]).reset_index(drop=True)
    
    # 兼容旧版 CSV (如果没有 Slope_abs)
    if "Slope_abs" not in df.columns:
        df["Slope_abs"] = df["Slope"].abs() if "Slope" in df.columns else 0.0

    torpor_selection = {}

    # --- 循环处理 GMM 和 HMM ---
    for label_col, prefix in [("label_gmm", "GMM"), ("label_hmm", "HMM")]:
        if label_col not in df.columns: continue
        
        print(f"\n--- Analyzing {prefix} ({label_col}) ---")

        # 1. 统计特征 (加入 P10)
        stats = df.groupby(label_col).agg(
            n_samples=("timestamp", "size"),
            T_rel_mean=("T_rel", "mean"),
            T_rel_median=("T_rel", "median"),
            T_rel_p10=("T_rel", lambda x: np.quantile(x, 0.1)), # <--- 关键升级
            Slope_abs_mean=("Slope_abs", "mean")
        ).reset_index().rename(columns={label_col: "cluster"})

        # 2. 统计时长
        bout_df, bout_agg = compute_bout_stats(df, label_col)

        # 3. 评分
        scored = score_torpor_candidates_v2(stats, bout_agg)
        
        # 4. 决策
        winner = scored.iloc[0]
        cid = int(winner["cluster"])
        
        # 安全检查: 中位体温不能太高
        if winner["T_rel_median"] > -0.5:
            print(f"[WARN] Cluster {cid} is too warm (Median={winner['T_rel_median']:.1f}). No Torpor detected.")
            torpor_cluster = None
        else:
            print(f"[RESULT] Torpor Identified: Cluster {cid}")
            print(f"         > Score: {winner['torpor_score']:.2f}")
            print(f"         > T_rel P10:    {winner['T_rel_p10']:.2f}°C (Deepest 10%)")
            print(f"         > T_rel Median: {winner['T_rel_median']:.2f}°C")
            torpor_cluster = cid

        # 5. 保存结果
        scored.to_csv(out_dir / f"{prefix}_ranking.csv", index=False)
        plot_diagnostics(df, bout_df, label_col, out_dir / "plots", prefix)
        
        # 6. 打标
        torpor_selection[prefix] = torpor_cluster
        flag_col = f"is_torpor_{prefix}"
        df[flag_col] = (df[label_col] == torpor_cluster).astype(int) if torpor_cluster is not None else 0

    # 保存最终 CSV
    out_csv = out_dir / "clusters_with_torpor_flag.csv"
    df.to_csv(out_csv, index=False)
    
    # 保存 JSON 配置
    with open(out_dir / "torpor_selection.json", "w") as f:
        json.dump(torpor_selection, f, indent=2)

    print(f"\n[DONE] Saved to: {out_csv}")

if __name__ == "__main__":
    main()