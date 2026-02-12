# 04_run_svm_kmeans_oneclick.py
# ------------------------------------------------------------
# One-click KMeans SVM (detection + early-warning prediction)
# - VSCode friendly: run this file directly.
# - Minimal progress logs (no spam).
# - GroupKFold by mouse for tuning (PR-AUC).
# - Clear outputs: metrics + curves + per-mouse timeline plots.
#
# Fix vs your crash:
#   - mouse_id can be NON-numeric (e.g., "N1"), so DO NOT coerce to numeric.
# ------------------------------------------------------------

from __future__ import annotations

import os
import json
import time
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    f1_score,
    confusion_matrix,
    classification_report,
)
import joblib


# =========================
# Config
# =========================
@dataclass
class Config:
    input_path: str = r"C:\Users\mayue\Desktop\Wavelet\out_cluster\kmeans\fixed_dynamic_clusters.xlsx"
    out_dir: str = r"C:\Users\mayue\Desktop\Wavelet\out_cluster\svm\svm_kmeans_clean"

    dayfirst: bool = False

    resample_rule: str = "5min"   # "5min" or "none"
    baseline_hours: int = 24
    min_periods: int = 12

    horizon_minutes: int = 30

    seed: int = 42
    n_test_mice: int = 3
    cv_splits: int = 3

    C_grid: Tuple[float, ...] = (1.0, 3.0)
    gamma_grid: Tuple[str, ...] = ("scale",)

    label_col: str = "cluster"
    torpor_cluster: int = 1
    torpor_flag_col: str = "is_torpor_KMEANS"

    plot_all_test_mice: bool = True


FEATURE_COLS: Tuple[str, ...] = (
    "T_rel", "Lag15", "Lag30", "Slope30", "Accel30", "hod_sin", "hod_cos"
)


# =========================
# Logger / utils
# =========================
def log(msg: str) -> None:
    t = time.strftime("%H:%M:%S")
    print(f"[{t}] {msg}", flush=True)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_name(x: object) -> str:
    s = str(x)
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", s)


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    suf = path.suffix.lower()
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suf in (".csv", ".txt"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file: {suf}")


def normalize_columns_kmeans(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename = {}

    if "mouse_id" not in df.columns:
        for c in ("mouse", "Mouse", "mouseID", "mouseId"):
            if c in df.columns:
                rename[c] = "mouse_id"
                break

    if "timestamp" not in df.columns:
        for c in ("Time", "time", "DateTime", "Datetime", "Timestamp"):
            if c in df.columns:
                rename[c] = "timestamp"
                break

    if "T" not in df.columns:
        for c in ("Temperature", "Temp", "Tb", "temp"):
            if c in df.columns:
                rename[c] = "T"
                break

    if "cluster" not in df.columns:
        for c in ("Cluster", "label", "label_kmeans", "kmeans_cluster"):
            if c in df.columns:
                rename[c] = "cluster"
                break

    return df.rename(columns=rename)


def parse_timestamp(s: pd.Series, dayfirst: bool) -> pd.Series:
    t = pd.to_datetime(s, errors="coerce", dayfirst=dayfirst)
    num = pd.to_numeric(s, errors="coerce")

    m_excel = t.isna() & num.notna() & (num > 30000) & (num < 80000)
    if m_excel.any():
        t.loc[m_excel] = pd.to_datetime(num[m_excel], unit="D", origin="1899-12-30", errors="coerce")

    m_s = t.isna() & num.notna() & (num > 1e9) & (num < 1e11)
    if m_s.any():
        t.loc[m_s] = pd.to_datetime(num[m_s], unit="s", errors="coerce")

    m_ms = t.isna() & num.notna() & (num > 1e11) & (num < 1e14)
    if m_ms.any():
        t.loc[m_ms] = pd.to_datetime(num[m_ms], unit="ms", errors="coerce")

    return t


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float("nan") if len(np.unique(y)) < 2 else float(roc_auc_score(y, p))


def safe_ap(y: np.ndarray, p: np.ndarray) -> float:
    return float("nan") if len(np.unique(y)) < 2 else float(average_precision_score(y, p))


def best_threshold_f1(y: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    if len(np.unique(y)) < 2:
        return 0.5, float("nan")
    ths = np.linspace(0.01, 0.99, 99)
    f1s = [f1_score(y, (p >= t).astype(int)) for t in ths]
    k = int(np.nanargmax(f1s))
    return float(ths[k]), float(f1s[k])


def resample_rule(cfg: Config) -> Optional[str]:
    r = cfg.resample_rule.strip().lower()
    return None if r in ("none", "off", "false", "0", "") else cfg.resample_rule


# =========================
# Torpor flag (KMeans)
# =========================
def attach_torpor_flag_kmeans(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, Dict[str, object]]:
    df = df.copy()
    info: Dict[str, object] = {"torpor_flag_col": cfg.torpor_flag_col, "method": "KMEANS"}

    if cfg.torpor_flag_col in df.columns:
        df[cfg.torpor_flag_col] = (
            pd.to_numeric(df[cfg.torpor_flag_col], errors="coerce")
            .fillna(0).astype(int).clip(0, 1)
        )
        info["source"] = "existing_column"
        return df, info

    if cfg.label_col not in df.columns:
        raise ValueError(f"Missing KMeans label column: {cfg.label_col}")

    df[cfg.label_col] = pd.to_numeric(df[cfg.label_col], errors="coerce")
    df = df.dropna(subset=[cfg.label_col]).copy()
    df[cfg.label_col] = df[cfg.label_col].astype(int)

    df[cfg.torpor_flag_col] = (df[cfg.label_col] == int(cfg.torpor_cluster)).astype(int)
    info["source"] = "cluster==torpor_cluster"
    info["torpor_cluster"] = int(cfg.torpor_cluster)
    info["label_col"] = cfg.label_col
    return df, info


# =========================
# Feature engineering
# =========================
def compute_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rule = resample_rule(cfg)

    df = df.copy()
    # IMPORTANT: mouse_id can be string like "N1"
    df["mouse_id"] = df["mouse_id"].astype(str)
    df["T"] = pd.to_numeric(df["T"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df = df.dropna(subset=["mouse_id", "timestamp", "T"]).copy()
    df = df.sort_values(["mouse_id", "timestamp"]).reset_index(drop=True)

    base_step_min = 5.0
    steps_15 = int(round(15.0 / base_step_min))
    steps_30 = int(round(30.0 / base_step_min))

    out: List[pd.DataFrame] = []
    for mid, sub in df.groupby("mouse_id"):
        sub = sub.set_index("timestamp").sort_index()

        if rule:
            sub = sub.resample(rule).median(numeric_only=True)
            sub["T"] = sub["T"].interpolate(limit=6)
            if cfg.torpor_flag_col in sub.columns:
                sub[cfg.torpor_flag_col] = (
                    sub[cfg.torpor_flag_col].interpolate(limit=6).round().clip(0, 1)
                )

        roll = sub["T"].rolling(
            window=f"{cfg.baseline_hours}h",
            min_periods=cfg.min_periods,
            closed="left",
        )
        sub["T_base"] = roll.median()
        sub["T_base"] = sub["T_base"].fillna(sub["T"].expanding().median())
        sub["T_rel"] = sub["T"] - sub["T_base"]

        sub["Lag15"] = sub["T_rel"].shift(steps_15)
        sub["Lag30"] = sub["T_rel"].shift(steps_30)
        sub["Slope30"] = sub["T_rel"] - sub["T_rel"].shift(steps_30)
        sub["Accel30"] = sub["Slope30"] - sub["Slope30"].shift(steps_30)

        hour = sub.index.hour + sub.index.minute / 60.0
        sub["hod_sin"] = np.sin(2 * np.pi * hour / 24.0)
        sub["hod_cos"] = np.cos(2 * np.pi * hour / 24.0)

        sub = sub.reset_index()
        sub["mouse_id"] = str(mid)
        out.append(sub)

    if not out:
        raise ValueError("No valid groups found after cleaning. Check mouse_id/timestamp/T parsing.")

    feat = pd.concat(out, ignore_index=True)

    need = ["mouse_id", "timestamp", "T", "T_base", cfg.torpor_flag_col] + list(FEATURE_COLS)
    missing = [c for c in need if c not in feat.columns]
    if missing:
        raise ValueError(f"Feature table missing columns: {missing}")

    feat = feat.dropna(subset=need).copy()
    feat[cfg.torpor_flag_col] = (
        pd.to_numeric(feat[cfg.torpor_flag_col], errors="coerce")
        .fillna(0).astype(int).clip(0, 1)
    )
    return feat


def build_targets(feat: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rule = resample_rule(cfg)
    base_step_min = 5.0 if rule else 5.0
    steps_future = int(round(cfg.horizon_minutes / base_step_min))

    detect = feat.copy()
    detect["y"] = detect[cfg.torpor_flag_col].astype(int)

    parts = []
    for mid, sub in feat.groupby("mouse_id"):
        sub = sub.sort_values("timestamp").copy()
        sub["y"] = sub[cfg.torpor_flag_col].shift(-steps_future)
        parts.append(sub)
    pred = pd.concat(parts, ignore_index=True)
    pred = pred.dropna(subset=["y"]).copy()
    pred["y"] = pd.to_numeric(pred["y"], errors="coerce").fillna(0).astype(int).clip(0, 1)

    return detect, pred


# =========================
# Model / tuning
# =========================
def make_model(C: float, gamma: str, seed: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=float(C),
            gamma=str(gamma),
            probability=True,
            class_weight="balanced",
            random_state=int(seed),
        )),
    ])


def split_mice(mice: np.ndarray, n_test: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    mice = np.array(sorted(pd.unique(mice.astype(str))), dtype=object)
    rng = np.random.RandomState(seed)
    rng.shuffle(mice)
    n_test = max(1, min(int(n_test), len(mice) - 1))
    return mice[n_test:], mice[:n_test]


def tune_hyperparams(X: np.ndarray, y: np.ndarray, groups: np.ndarray, cfg: Config) -> Tuple[float, str, float]:
    n_groups = len(np.unique(groups))
    n_splits = min(cfg.cv_splits, n_groups) if n_groups >= 2 else 2
    gkf = GroupKFold(n_splits=n_splits)

    best_score = -np.inf
    best_C, best_gamma = cfg.C_grid[0], cfg.gamma_grid[0]

    total = len(cfg.C_grid) * len(cfg.gamma_grid)
    done = 0

    for C in cfg.C_grid:
        for gamma in cfg.gamma_grid:
            done += 1
            log(f"  CV tuning {done}/{total}: C={C}, gamma={gamma}")

            scores: List[float] = []
            for tr, va in gkf.split(X, y, groups):
                model = make_model(C, gamma, cfg.seed)
                model.fit(X[tr], y[tr])
                p = model.predict_proba(X[va])[:, 1]
                scores.append(safe_ap(y[va], p))

            m = float(np.nanmean(scores)) if scores else float("nan")
            if np.isfinite(m) and m > best_score:
                best_score, best_C, best_gamma = m, float(C), str(gamma)

    return float(best_C), str(best_gamma), float(best_score)


# =========================
# Plotting
# =========================
def plot_mouse_timeline(df: pd.DataFrame, thr: float, title: str, save_path: Path) -> None:
    if df.empty:
        return

    t = df["timestamp"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    ax1.plot(t, df["T"], linewidth=1.0, alpha=0.45, label="T")
    ax1.plot(t, df["T_base"], linewidth=1.2, alpha=0.95, label="T_base (past median)")
    ax1.set_ylabel("Temperature (°C)")
    ax1.legend(loc="upper left")

    ax2.plot(t, df["prob"], linewidth=1.5, label="SVM prob")
    ax2.axhline(thr, linestyle="--", linewidth=1.0, label=f"thr={thr:.2f}")
    ax2.fill_between(t, 0, 1, where=(df["y"].to_numpy(int) == 1), alpha=0.18, label="True y=1")
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_ylabel("Probability")
    ax2.legend(loc="upper left")

    fig.suptitle(title, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=170)
    plt.close()


def plot_curves(y: np.ndarray, p: np.ndarray, out_dir: Path, prefix: str) -> None:
    if len(np.unique(y)) < 2:
        return

    prec, rec, _ = precision_recall_curve(y, p)
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR (test) - {prefix}")
    plt.tight_layout()
    plt.savefig(out_dir / f"pr_curve_test_{prefix}.png", dpi=170)
    plt.close()

    fpr, tpr, _ = roc_curve(y, p)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC (test) - {prefix}")
    plt.tight_layout()
    plt.savefig(out_dir / f"roc_curve_test_{prefix}.png", dpi=170)
    plt.close()


# =========================
# One task runner
# =========================
def run_task(df: pd.DataFrame, task_name: str, out_dir: Path, cfg: Config) -> Dict[str, object]:
    log(f"[{task_name}] Split train/test mice")
    train_mice, test_mice = split_mice(df["mouse_id"].to_numpy(str), cfg.n_test_mice, cfg.seed)

    train = df[df["mouse_id"].isin(train_mice)].copy()
    test = df[df["mouse_id"].isin(test_mice)].copy()

    X_train = train[list(FEATURE_COLS)].to_numpy(float)
    y_train = train["y"].to_numpy(int)
    g_train = train["mouse_id"].to_numpy(str)

    X_test = test[list(FEATURE_COLS)].to_numpy(float)
    y_test = test["y"].to_numpy(int)

    log(f"[{task_name}] CV tuning (GroupKFold by mouse) ...")
    best_C, best_gamma, cv_pr_auc = tune_hyperparams(X_train, y_train, g_train, cfg)
    log(f"[{task_name}] Best params: C={best_C}, gamma={best_gamma}, cv_pr_auc={cv_pr_auc:.4f}")

    log(f"[{task_name}] Train final model")
    model = make_model(best_C, best_gamma, cfg.seed)
    model.fit(X_train, y_train)

    p_train = model.predict_proba(X_train)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]

    thr, f1_tr = best_threshold_f1(y_train, p_train)
    pred05 = (p_test >= 0.5).astype(int)
    predth = (p_test >= thr).astype(int)

    met = {
        "task": task_name,
        "train_mice": list(map(str, train_mice)),
        "test_mice": list(map(str, test_mice)),
        "best_C": best_C,
        "best_gamma": best_gamma,
        "cv_mean_pr_auc": cv_pr_auc,
        "thr_f1opt_train": thr,
        "f1_train_best": f1_tr,
        "roc_auc_test": safe_auc(y_test, p_test),
        "ap_test": safe_ap(y_test, p_test),
        "f1_test_thr0.5": float(f1_score(y_test, pred05)) if len(np.unique(y_test)) >= 2 else float("nan"),
        "f1_test_thr_f1opt": float(f1_score(y_test, predth)) if len(np.unique(y_test)) >= 2 else float("nan"),
    }

    cm05 = confusion_matrix(y_test, pred05, labels=[0, 1]).ravel()
    cmth = confusion_matrix(y_test, predth, labels=[0, 1]).ravel()
    met.update({f"cm05_{k}": int(v) for k, v in zip(("tn", "fp", "fn", "tp"), cm05)})
    met.update({f"cmth_{k}": int(v) for k, v in zip(("tn", "fp", "fn", "tp"), cmth)})

    joblib.dump(model, out_dir / f"model_{task_name}.joblib")

    (out_dir / f"{task_name}_report_thr0.5.txt").write_text(
        classification_report(y_test, pred05, digits=4), encoding="utf-8"
    )
    (out_dir / f"{task_name}_report_thr_f1opt.txt").write_text(
        f"thr={thr:.6f}\n\n{classification_report(y_test, predth, digits=4)}",
        encoding="utf-8"
    )

    test_out = test.copy()
    test_out["prob"] = p_test
    test_out["pred_thr0.5"] = pred05
    test_out["pred_thr_f1opt"] = predth
    test_out.to_csv(out_dir / f"test_predictions_{task_name}.csv", index=False)

    plot_curves(y_test, p_test, out_dir, task_name)

    log(f"[{task_name}] Plot timelines for test mice")
    plot_mice = list(test_mice) if cfg.plot_all_test_mice else [str(test_mice[0])]
    for mid in plot_mice:
        sub = test_out[test_out["mouse_id"] == mid].sort_values("timestamp")
        plot_mouse_timeline(
            sub,
            thr=thr,
            title=f"Mouse {mid} | {task_name} | test",
            save_path=out_dir / f"Mouse_{safe_name(mid)}_{task_name}.png",
        )

    return met


# =========================
# Main
# =========================
def main(cfg: Config) -> None:
    out_dir = Path(cfg.out_dir)
    ensure_dir(out_dir)

    log("Step 1/5: Load data")
    df = read_table(Path(cfg.input_path))
    df = normalize_columns_kmeans(df)

    required = ("mouse_id", "timestamp", "T", "cluster")
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df["timestamp"] = parse_timestamp(df["timestamp"], cfg.dayfirst)
    df = df.dropna(subset=["timestamp"]).copy()
    df["mouse_id"] = df["mouse_id"].astype(str)

    log(f"  Rows={len(df):,}, mice={df['mouse_id'].nunique()}")

    log("Step 2/5: Attach torpor flag (KMeans)")
    df, torpor_info = attach_torpor_flag_kmeans(df, cfg)
    (out_dir / "torpor_selection_kmeans.json").write_text(json.dumps(torpor_info, indent=2), encoding="utf-8")
    log(f"  Torpor flag source={torpor_info.get('source')}, torpor_cluster={torpor_info.get('torpor_cluster', 'NA')}")

    log("Step 3/5: Feature engineering (past-only baseline)")
    feat = compute_features(df[["mouse_id", "timestamp", "T", "cluster", cfg.torpor_flag_col]].copy(), cfg)
    feat.to_csv(out_dir / "features_all.csv", index=False)
    log(f"  Feature rows={len(feat):,}")

    log("Step 4/5: Build detection/prediction datasets")
    detect, pred = build_targets(feat, cfg)
    detect.to_csv(out_dir / "dataset_detect.csv", index=False)
    pred.to_csv(out_dir / "dataset_predict.csv", index=False)
    log(f"  detect={len(detect):,}, predict={len(pred):,} (horizon={cfg.horizon_minutes}min)")

    log("Step 5/5: Train/Evaluate models")
    met_detect = run_task(detect, "detect", out_dir, cfg)
    met_pred = run_task(pred, "predict", out_dir, cfg)

    metrics = {"config": asdict(cfg), "torpor_info": torpor_info, "detect": met_detect, "predict": met_pred}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame([met_detect, met_pred]).to_csv(out_dir / "metrics.csv", index=False)

    log(f"DONE. Outputs -> {out_dir.resolve()}")


if __name__ == "__main__":
    cfg = Config(
        input_path=os.environ.get("KMEANS_INPUT", Config.input_path),
        out_dir=os.environ.get("KMEANS_OUTDIR", Config.out_dir),
    )
    main(cfg)
