import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# -----------------------------
# Default Paths (MATCH YOUR FOLDER)
# -----------------------------
BASE = Path(r"C:\Users\mayue\Desktop\Wavelet\out_cluster")
DEFAULT_ID_DIR = BASE / "out_torpor_identification"
DEFAULT_MARKED_CSV = DEFAULT_ID_DIR / "clusters_with_torpor_flag.csv"
DEFAULT_SEL_JSON = DEFAULT_ID_DIR / "torpor_selection.json"
DEFAULT_OUT_DIR = BASE / "out_svm_from_clusters"


# -----------------------------
# Utils
# -----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def parse_timestamp_series(s: pd.Series, dayfirst: bool = False) -> pd.Series:
    """
    Robust timestamp parser:
    - strings
    - Excel serial (days since 1899-12-30)
    - Unix seconds
    - Unix milliseconds
    """
    t = pd.to_datetime(s, errors="coerce", dayfirst=dayfirst)
    num = pd.to_numeric(s, errors="coerce")

    # Excel serial
    m_excel = t.isna() & num.notna() & (num > 30000) & (num < 80000)
    if m_excel.any():
        t.loc[m_excel] = pd.to_datetime(num[m_excel], unit="D", origin="1899-12-30", errors="coerce")

    # Unix seconds
    m_unix_s = t.isna() & num.notna() & (num > 1e9) & (num < 1e11)
    if m_unix_s.any():
        t.loc[m_unix_s] = pd.to_datetime(num[m_unix_s], unit="s", errors="coerce")

    # Unix ms
    m_unix_ms = t.isna() & num.notna() & (num > 1e11) & (num < 1e14)
    if m_unix_ms.any():
        t.loc[m_unix_ms] = pd.to_datetime(num[m_unix_ms], unit="ms", errors="coerce")

    return t


def median_dt_seconds(t: pd.Series) -> float:
    dt = t.sort_values().diff().dt.total_seconds().median()
    if not np.isfinite(dt) or dt <= 0:
        return 300.0  # fallback 5 min
    return float(dt)


def rolling_mean_by_time_index(x: pd.Series, t: pd.Series, window_minutes: float) -> pd.Series:
    dt_sec = median_dt_seconds(t)
    win = int(max(1, round((window_minutes * 60.0) / dt_sec)))
    return x.rolling(win, center=True, min_periods=max(1, win // 3)).mean()


def spans_bool(mask: np.ndarray):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    out = []
    s = p = idx[0]
    for i in idx[1:]:
        if i == p + 1:
            p = i
        else:
            out.append((s, p))
            s = p = i
    out.append((s, p))
    return out


def choose_thresholds(prob_ref: np.ndarray, y_ref: np.ndarray):
    """
    Stable shading thresholds.
    If single-class -> fixed thresholds.
    Else:
      gray_thr = 99.5% quantile of negatives
      deep_thr = 50% quantile of positives
    """
    prob_ref = np.asarray(prob_ref, float)
    y_ref = np.asarray(y_ref, int)

    if len(np.unique(y_ref)) < 2:
        return 0.8, 0.2

    neg = prob_ref[y_ref == 0]
    pos = prob_ref[y_ref == 1]

    gray_thr = float(np.quantile(neg, 0.995)) if len(neg) >= 20 else 0.2
    deep_thr = float(np.quantile(pos, 0.50)) if len(pos) >= 20 else 0.8

    if deep_thr <= gray_thr:
        deep_thr = float(min(1.0, max(gray_thr + 0.05, 0.8)))

    return float(np.clip(deep_thr, 0, 1)), float(np.clip(gray_thr, 0, 1))


# -----------------------------
# Scheme A split (robust): force 1 positive mouse per split if >=3 pos mice
# -----------------------------
def split_mice_force_pos_each(work: pd.DataFrame, y: np.ndarray, group_col: str, seed: int):
    groups = work[group_col].values
    mouse_df = pd.DataFrame({"mouse_id": groups, "y": y})
    mouse_y = mouse_df.groupby("mouse_id")["y"].max().astype(int)

    mouse_ids = mouse_y.index.values
    mouse_labels = mouse_y.values
    pos_mice = mouse_ids[mouse_labels == 1].tolist()
    neg_mice = mouse_ids[mouse_labels == 0].tolist()

    rng = np.random.RandomState(seed)
    rng.shuffle(pos_mice)
    rng.shuffle(neg_mice)

    n_pos = len(pos_mice)
    n_neg = len(neg_mice)

    if n_pos < 2:
        raise RuntimeError(
            f"Not enough positive mice: pos_mice={n_pos}, neg_mice={n_neg}. "
            f"Need >=2 pos mice for meaningful evaluation."
        )

    # If we have >=3 positive mice, guarantee train/val/test each has positives
    if n_pos >= 3:
        train_pos = [pos_mice[0]]
        val_pos = [pos_mice[1]]
        test_pos = [pos_mice[2]]

        n_total = len(mouse_ids)
        n_test = max(1, int(round(0.25 * n_total)))
        n_trainval = n_total - n_test
        n_val = max(1, int(round(0.25 * n_trainval)))

        test_mice = set(test_pos + neg_mice[: max(0, n_test - 1)])
        remaining_neg = neg_mice[max(0, n_test - 1):]

        val_mice = set(val_pos + remaining_neg[: max(0, n_val - 1)])
        remaining_neg = remaining_neg[max(0, n_val - 1):]

        train_mice = set(train_pos + remaining_neg + pos_mice[3:])
        return train_mice, val_mice, test_mice, n_pos, n_neg

    # Only 2 pos mice: train gets one, test gets one, val may have no positives
    train_pos = [pos_mice[0]]
    test_pos = [pos_mice[1]]

    n_total = len(mouse_ids)
    n_test = max(1, int(round(0.25 * n_total)))
    test_mice = set(test_pos + neg_mice[: max(0, n_test - 1)])
    remaining_neg = neg_mice[max(0, n_test - 1):]

    n_val = max(1, int(round(0.25 * (n_total - len(test_mice)))))
    val_mice = set(remaining_neg[:n_val])
    train_mice = set(train_pos + remaining_neg[n_val:])

    return train_mice, val_mice, test_mice, n_pos, n_neg


# -----------------------------
# Plot Torpor Map
# -----------------------------
def plot_torpor_map_one_mouse(
    sub: pd.DataFrame,
    out_png: Path,
    title: str,
    tb_col: str,
    prob_raw_col: str,
    prob_smooth_col: str,
    fasting_start: pd.Timestamp | None,
    deep_thr: float,
    gray_thr: float,
    drop_first_days: float,
):
    sub = sub.sort_values("timestamp").copy()
    if drop_first_days and drop_first_days > 0:
        t0 = sub["timestamp"].min() + pd.Timedelta(days=float(drop_first_days))
        sub = sub[sub["timestamp"] >= t0].copy()
        if sub.empty:
            return

    t = sub["timestamp"]
    tb = sub[tb_col].astype(float)
    tb_s = rolling_mean_by_time_index(tb, t, window_minutes=20)

    pr = sub[prob_raw_col].astype(float).clip(0, 1)
    ps = sub[prob_smooth_col].astype(float).clip(0, 1)

    deep_mask = (ps.values >= deep_thr)
    gray_mask = (ps.values >= gray_thr) & (ps.values < deep_thr)

    deep_spans = spans_bool(deep_mask)
    gray_spans = spans_bool(gray_mask)

    fig = plt.figure(figsize=(20, 7), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.12)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, tb.values, linewidth=1.0, label="Tb")
    ax1.plot(t, tb_s.values, linewidth=1.2, label="Tb smooth (20min)")
    if fasting_start is not None and pd.notna(fasting_start):
        ax1.axvline(fasting_start, linestyle="--", linewidth=1.5, label="Fasting start")

    for a, b in gray_spans:
        ax1.axvspan(t.iloc[a], t.iloc[b], alpha=0.10)
    for a, b in deep_spans:
        ax1.axvspan(t.iloc[a], t.iloc[b], alpha=0.15)

    ax1.set_ylabel("Tb (°C)")
    ax1.set_title(title)
    ax1.legend(loc="upper right")

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.plot(t, pr.values, linewidth=1.0, label="SVM prob (raw)")
    ax2.plot(t, ps.values, linewidth=1.2, label="SVM prob (smooth)")
    ax2.axhline(deep_thr, linestyle=":", linewidth=1.5, label=f"deep_thr={deep_thr:.4f}")
    ax2.axhline(gray_thr, linestyle=":", linewidth=1.5, label=f"gray_thr={gray_thr:.4f}")

    for a, b in gray_spans:
        ax2.axvspan(t.iloc[a], t.iloc[b], alpha=0.10)
    for a, b in deep_spans:
        ax2.axvspan(t.iloc[a], t.iloc[b], alpha=0.15)

    ax2.set_ylabel("Probability")
    ax2.set_xlabel("Time")
    ax2.set_ylim(-0.02, 1.02)
    ax2.legend(loc="upper right")

    ensure_dir(out_png.parent)
    plt.savefig(out_png, dpi=160)
    plt.close()


# -----------------------------
# Train + Eval + Plot
# -----------------------------
def run_one(
    df: pd.DataFrame,
    label_col: str,          # is_torpor_GMM / is_torpor_HMM
    tag: str,                # "GMM" / "HMM"
    out_dir: Path,
    feature_cols: list[str],
    group_col: str,
    seed: int,
    do_plots: bool,
    fasting_start_str: str | None,
    drop_first_days: float,
    keep_neg_ratio: float,
    time_clip_lo: float,
    time_clip_hi: float,
):
    if label_col not in df.columns:
        print(f"[WARN] Missing {label_col}, skip {tag}.")
        return None

    work = df.dropna(subset=feature_cols + [label_col, group_col, "timestamp"]).copy()
    work[label_col] = work[label_col].astype(int)

    # clip extreme timestamps so axis doesn't explode
    if 0 <= time_clip_lo < time_clip_hi <= 1:
        lo, hi = work["timestamp"].quantile(time_clip_lo), work["timestamp"].quantile(time_clip_hi)
        work = work[(work["timestamp"] >= lo) & (work["timestamp"] <= hi)].copy()

    pos = work[work[label_col] == 1]
    neg = work[work[label_col] == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise RuntimeError(f"{tag}: No pos/neg samples. Check 02 output labels.")

    # downsample neg
    max_neg = int(min(len(neg), int(round(keep_neg_ratio * len(pos)))))
    neg = neg.sample(n=max_neg, random_state=seed)
    work = pd.concat([pos, neg], ignore_index=True).sample(frac=1.0, random_state=seed)

    X = work[feature_cols].astype(float)
    y = work[label_col].values.astype(int)

    train_mice, val_mice, test_mice, n_pos_mice, n_neg_mice = split_mice_force_pos_each(
        work=work, y=y, group_col=group_col, seed=seed
    )

    idx_all = np.arange(len(work))
    m_all = work[group_col].values
    tr_idx = idx_all[np.isin(m_all, list(train_mice))]
    va_idx = idx_all[np.isin(m_all, list(val_mice))]
    te_idx = idx_all[np.isin(m_all, list(test_mice))]

    X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
    X_va, y_va = X.iloc[va_idx], y[va_idx]
    X_te, y_te = X.iloc[te_idx], y[te_idx]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", C=1.5, class_weight="balanced", probability=True, random_state=seed)),
    ])
    model.fit(X_tr, y_tr)

    def eval_split(name, Xs, ys):
        prob = model.predict_proba(Xs)[:, 1]
        pred = (prob >= 0.5).astype(int)
        cm = confusion_matrix(ys, pred, labels=[0, 1])
        auc = roc_auc_score(ys, prob) if len(np.unique(ys)) > 1 else float("nan")
        ap = average_precision_score(ys, prob) if len(np.unique(ys)) > 1 else float("nan")
        f1 = f1_score(ys, pred) if len(np.unique(ys)) > 1 else float("nan")
        return {"split": name, "auc": float(auc), "ap": float(ap), "f1": float(f1), "cm": cm, "prob": prob, "pred": pred}

    ev_tr = eval_split("train", X_tr, y_tr)
    ev_va = eval_split("val", X_va, y_va)
    ev_te = eval_split("test", X_te, y_te)

    deep_thr, gray_thr = choose_thresholds(ev_tr["prob"], y_tr)

    # save metrics
    rows = []
    for ev in [ev_tr, ev_va, ev_te]:
        cm = ev["cm"]
        rows.append({
            "tag": tag,
            "split": ev["split"],
            "auc": ev["auc"],
            "ap": ev["ap"],
            "f1": ev["f1"],
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        })
    met = pd.DataFrame(rows)
    met.to_csv(out_dir / f"svm_{tag}_metrics.csv", index=False)

    with open(out_dir / f"svm_{tag}_classification_report.txt", "w", encoding="utf-8") as f:
        f.write("=== SPLIT INFO (mice) ===\n")
        f.write(f"train_mice={len(train_mice)} val_mice={len(val_mice)} test_mice={len(test_mice)}\n")
        f.write(f"pos_mice_total={n_pos_mice} neg_mice_total={n_neg_mice}\n\n")
        f.write("=== TRAIN ===\n")
        f.write(classification_report(y_tr, (ev_tr["prob"] >= 0.5).astype(int), digits=4))
        f.write("\n\n=== VAL ===\n")
        f.write(classification_report(y_va, (ev_va["prob"] >= 0.5).astype(int), digits=4))
        f.write("\n\n=== TEST ===\n")
        f.write(classification_report(y_te, (ev_te["prob"] >= 0.5).astype(int), digits=4))

    # full predictions for plots
    full = df.dropna(subset=feature_cols + [group_col, "timestamp"]).copy()
    if 0 <= time_clip_lo < time_clip_hi <= 1:
        lo, hi = full["timestamp"].quantile(time_clip_lo), full["timestamp"].quantile(time_clip_hi)
        full = full[(full["timestamp"] >= lo) & (full["timestamp"] <= hi)].copy()

    prob_col = f"svm_prob_{tag}"
    smooth_col = f"svm_prob_smooth_{tag}"
    full[prob_col] = model.predict_proba(full[feature_cols].astype(float))[:, 1]
    full[smooth_col] = np.nan

    for mid, sub in full.groupby(group_col):
        sub = sub.sort_values("timestamp")
        s = pd.Series(sub[prob_col].values, index=sub.index)
        sm = rolling_mean_by_time_index(s, sub["timestamp"], window_minutes=20)
        full.loc[sub.index, smooth_col] = sm.values

    full.to_csv(out_dir / f"svm_{tag}_full_predictions.csv", index=False)

    thr = {
        "tag": tag,
        "label_col": label_col,
        "deep_thr": deep_thr,
        "gray_thr": gray_thr,
        "fasting_start": fasting_start_str,
        "drop_first_days": drop_first_days,
        "feature_cols": list(feature_cols),
        "time_clip": [time_clip_lo, time_clip_hi],
        "keep_neg_ratio": keep_neg_ratio,
    }
    with open(out_dir / f"svm_{tag}_thresholds.json", "w", encoding="utf-8") as f:
        json.dump(thr, f, indent=2)

    # plots (DEFAULT ON)
    if do_plots:
        tb_col = "Tb" if "Tb" in full.columns else ("T" if "T" in full.columns else None)
        if tb_col is None:
            print(f"[WARN] No Tb/T column found; skip Torpor Maps for {tag}.")
        else:
            plots_dir = out_dir / f"plots_torpor_map_{tag}"
            ensure_dir(plots_dir)

            fasting_start = None
            if fasting_start_str:
                fasting_start = pd.to_datetime(fasting_start_str, errors="coerce")

            for mid in sorted(full[group_col].unique().tolist()):
                sub = full[full[group_col] == mid].copy()
                if sub.empty:
                    continue
                out_png = plots_dir / f"TorporMap_{mid}_{tag}.png"
                title = f"Torpor Map: {mid} (drop first {drop_first_days:g} days)"
                plot_torpor_map_one_mouse(
                    sub=sub,
                    out_png=out_png,
                    title=title,
                    tb_col=tb_col,
                    prob_raw_col=prob_col,
                    prob_smooth_col=smooth_col,
                    fasting_start=fasting_start,
                    deep_thr=deep_thr,
                    gray_thr=gray_thr,
                    drop_first_days=drop_first_days,
                )

    return met


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--marked_csv", type=str, default=str(DEFAULT_MARKED_CSV))
    ap.add_argument("--selection_json", type=str, default=str(DEFAULT_SEL_JSON))
    ap.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--dayfirst", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    # DEFAULT = PLOTS ON
    ap.add_argument("--no_plot_maps", action="store_true", help="Disable Torpor Map plotting")
    ap.add_argument("--fasting_start", type=str, default="2023-09-05 12:00:00")
    ap.add_argument("--drop_first_days", type=float, default=2.0)

    # robustness
    ap.add_argument("--keep_neg_ratio", type=float, default=5.0)
    ap.add_argument("--time_clip_lo", type=float, default=0.001)
    ap.add_argument("--time_clip_hi", type=float, default=0.999)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    marked_csv = Path(args.marked_csv)
    if not marked_csv.exists():
        raise FileNotFoundError(f"marked_csv not found: {marked_csv}")

    sel_json = Path(args.selection_json)
    if not sel_json.exists():
        print(f"[WARN] selection_json not found (not required): {sel_json}")

    df = pd.read_csv(marked_csv)

    if "timestamp" not in df.columns:
        raise RuntimeError("CSV must have 'timestamp' column.")
    if "mouse_id" not in df.columns:
        raise RuntimeError("CSV must have 'mouse_id' column.")

    df["timestamp"] = parse_timestamp_series(df["timestamp"], dayfirst=args.dayfirst)
    df = df.dropna(subset=["timestamp"]).copy()

    feature_cols = ["T_rel", "Slope", "T_rel_lag15", "T_rel_lag30"]
    for c in feature_cols:
        if c not in df.columns:
            raise RuntimeError(f"Missing feature column: {c}")

    label_map = {"GMM": "is_torpor_GMM", "HMM": "is_torpor_HMM"}

    all_metrics = []
    for tag, labcol in label_map.items():
        met = run_one(
            df=df,
            label_col=labcol,
            tag=tag,
            out_dir=out_dir,
            feature_cols=feature_cols,
            group_col="mouse_id",
            seed=args.seed,
            do_plots=(not args.no_plot_maps),
            fasting_start_str=(args.fasting_start if args.fasting_start else None),
            drop_first_days=float(args.drop_first_days),
            keep_neg_ratio=float(args.keep_neg_ratio),
            time_clip_lo=float(args.time_clip_lo),
            time_clip_hi=float(args.time_clip_hi),
        )
        if met is not None:
            all_metrics.append(met)

    if all_metrics:
        pd.concat(all_metrics, ignore_index=True).to_csv(out_dir / "svm_metrics_all.csv", index=False)

    print("[OK] Done. Output dir:", out_dir.resolve())


if __name__ == "__main__":
    main()
