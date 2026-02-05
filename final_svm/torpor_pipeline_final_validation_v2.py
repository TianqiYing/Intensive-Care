import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ============================================================
# Torpor pipeline (SVM + Oxford-style anchor) - FIXED version
#
# Fix: prevent ultra-low probability thresholds that cause "everything is Deep".
# We keep Oxford ref (Tb-based) as a physiological anchor, but thresholds must also
# respect the global probability distribution (global floors).
#
# Requirements preserved:
# 1) Drop first 2 days for N (cold-start).
# 2) Oxford key: 20min smoothing + median-2SD/3SD + Deep>=1h + Gray>=30min.
# 3) N-only validations + plots (probability axis always [0,1]).
# ============================================================


# ---------------- Paths ----------------
ROOT = Path(r"C:\Users\mayue\Desktop\Wavelet\Wavelet")
OUT  = Path(r"C:\Users\mayue\Desktop\Wavelet\out_oxford_svm_fixed")

F_DIR = ROOT / "Natural torpor HR, Core Temp, Activity"
N_DIR = ROOT / "Natural torpor N1-16"

# ---------------- Experiment meta ----------------
FASTING_START = pd.Timestamp("2023-09-05 12:00:00")
DROP_FIRST_DAYS_N = 2

# ---------------- Oxford-style parameters ----------------
ROLL_WIN = "20min"
K_GRAY, K_DEEP = 2.0, 3.0
MIN_DEEP_MIN, MIN_GRAY_MIN = 60, 30

# Baseline estimation for Oxford Tb thresholds:
# - median from top Tb subset (euthermic)
# - SD from trimmed band to capture normal variance but avoid torpor tail
MED_TOP_FRAC = 0.30
SD_TRIM_LO = 0.30
SD_TRIM_HI = 0.95
MIN_BASE_POINTS = 120

# ---------------- F weak-label settings ----------------
F_BASELINE_KEY = "4Nov"
TORPOR_ANCHOR_DAYS = {
    "22Oct2023.xlsx": ["F4", "F5", "F6"],
    "28Oct2023.xlsx": ["F3"],
    "18Nov2023.xlsx": ["F1", "F4", "F5"],
}
Z_ANCHOR = 2.0

# ---------------- Model / features ----------------
FEATS = ["Tb_diff", "Tb_slope", "HR_ratio", "hour_sin", "hour_cos"]
SVM_C = 1.5
SKIP_ROWS = 60

# ---------------- Probability threshold sweep ----------------
# q_deep/q_gray: ref-based quantiles (Oxford anchor)
# q_deep_global/q_gray_global: global floor quantiles (prevents degenerate low thresholds)
SWEEP = [
    # (q_deep_ref, q_gray_ref, q_deep_global, q_gray_global)
    (0.30, 0.80, 0.990, 0.950),
    (0.30, 0.80, 0.995, 0.960),
    (0.20, 0.70, 0.990, 0.950),
    (0.20, 0.70, 0.995, 0.960),
]

# Guardrails
MAX_DEEP_FRAC_12H = 0.02
MAX_GRAY_FRAC_12H_MEAN = 0.25


# ---------------- Utils ----------------
def ensure(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def dt_minutes(sub: pd.DataFrame) -> float:
    d = sub["datetime"].sort_values().diff().dropna()
    return float(d.median().total_seconds() / 60) if len(d) else 5.0


def rolling_time(sub: pd.DataFrame, col: str) -> pd.Series:
    return (
        sub.set_index("datetime")[col]
        .astype(float)
        .rolling(ROLL_WIN, min_periods=1)
        .mean()
        .reset_index(drop=True)
    )


def spans(mask: np.ndarray):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    out, s, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if i == p + 1:
            p = i
        else:
            out.append((s, p))
            s = p = i
    out.append((s, p))
    return out


def keep_min_duration(mask: np.ndarray, step_min: float, min_min: int) -> np.ndarray:
    L = max(1, int(np.ceil(min_min / step_min)))
    keep = np.zeros_like(mask, dtype=bool)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return keep
    s = p = idx[0]
    for i in idx[1:]:
        if i == p + 1:
            p = i
        else:
            if p - s + 1 >= L:
                keep[s : p + 1] = True
            s = p = i
    if p - s + 1 >= L:
        keep[s : p + 1] = True
    return keep


def robust_baseline_T(tb: pd.Series) -> float:
    x = tb[(tb > 33) & (tb < 42)].dropna()
    if x.empty:
        return 37.0
    q = x.quantile(0.85)
    return float(x[x >= q].mean())


# ---------------- Data loading ----------------
def load_F(folder: Path) -> pd.DataFrame:
    rows = []
    for f in folder.glob("*.xlsx"):
        head = pd.read_excel(f, nrows=25, header=None)
        h = 0
        for i, r in head.iterrows():
            if "Time Stamp" in " ".join(map(str, r.values)):
                h = i
                break
        df = pd.read_excel(f, header=h)
        df.columns = [str(c).replace("\n", " ").strip() for c in df.columns]
        tcol = next((c for c in df.columns if "Time" in c), None)
        if tcol is None:
            continue

        mice = set(re.findall(r"([FM]\d+)", " ".join(df.columns)))
        for mid in mice:
            mcols = [c for c in df.columns if c.startswith(mid)]
            if not mcols:
                continue
            sub = df[[tcol] + mcols].copy()
            sub.columns = ["datetime"] + [c.split(".")[-1].strip() for c in mcols]

            ren = {}
            for c in sub.columns:
                lc = str(c).lower()
                if "temp" in lc:
                    ren[c] = "Tb"
                elif "heart" in lc or lc == "hr":
                    ren[c] = "HR"
                elif "activity" in lc or "act" in lc:
                    ren[c] = "Act"
            sub = sub.rename(columns=ren)
            if "Tb" not in sub.columns:
                continue

            sub["mouse_id"] = mid
            sub["source_file"] = f.name
            rows.append(sub)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_N(folder: Path) -> pd.DataFrame:
    rows = []
    for f in folder.glob("*.xlsx"):
        df = pd.read_excel(f)
        cols = [str(c).lower() for c in df.columns]
        tcol = next((df.columns[i] for i, c in enumerate(cols) if "time" in c or "datetime" in c), None)
        tbcol = next((df.columns[i] for i, c in enumerate(cols) if "temp" in c or "temperature" in c or c == "tb"), None)
        if tcol is None or tbcol is None:
            continue

        sub = pd.DataFrame(
            {
                "datetime": pd.to_datetime(df[tcol], errors="coerce"),
                "Tb": pd.to_numeric(df[tbcol], errors="coerce"),
            }
        )
        sub["mouse_id"] = f.stem
        sub["source_file"] = f.name
        rows.append(sub)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ---------------- Preprocess + features ----------------
def preprocess(df: pd.DataFrame, is_N: bool) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])

    for c in ["Tb", "HR", "Act"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    out = []
    for (mid, src), sub in df.groupby(["mouse_id", "source_file"]):
        sub = sub.sort_values("datetime").reset_index(drop=True)

        if len(sub) > SKIP_ROWS:
            sub = sub.iloc[SKIP_ROWS:].copy()

        if is_N and DROP_FIRST_DAYS_N > 0 and not sub.empty:
            t0 = sub["datetime"].iloc[0]
            sub = sub[sub["datetime"] >= (t0 + pd.Timedelta(days=DROP_FIRST_DAYS_N))].copy().reset_index(drop=True)

        sub = sub[(sub["Tb"] > 28) & (sub["Tb"] < 42)].copy()
        if sub.empty:
            continue

        sub["Tb_slope"] = sub["Tb"].diff().rolling(20, min_periods=1).mean().fillna(0.0)
        base = robust_baseline_T(sub["Tb"])
        sub["Tb_diff"] = sub["Tb"] - base

        if "HR" in sub.columns and sub["HR"].notna().any():
            hb = sub[sub["Tb"] > (base - 1.2)]["HR"].median()
            hb = float(hb) if (hb is not None and np.isfinite(hb) and hb > 0) else 600.0
            sub["HR_ratio"] = sub["HR"] / hb
        else:
            sub["HR_ratio"] = 1.0

        h = sub["datetime"].dt.hour + sub["datetime"].dt.minute / 60.0
        sub["hour_sin"] = np.sin(2 * np.pi * (h - 12) / 24.0)
        sub["hour_cos"] = np.cos(2 * np.pi * (h - 12) / 24.0)

        out.append(sub)

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ---------------- Weak labels for F ----------------
def weak_label_F(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["label"] = -1
    df.loc[df["source_file"].str.contains(F_BASELINE_KEY, case=False, na=False), "label"] = 0

    for fname, mice in TORPOR_ANCHOR_DAYS.items():
        for mid in mice:
            m = (df["source_file"] == fname) & (df["mouse_id"] == mid)
            if not m.any():
                continue
            tb = df.loc[m, "Tb"]
            mu, sd = tb.mean(), tb.std()
            if not np.isfinite(sd) or sd <= 1e-6:
                continue
            thr = mu - Z_ANCHOR * sd
            df.loc[m & (df["Tb"] < thr), "label"] = 1

    return df


# ---------------- Oxford baseline (robust) ----------------
def oxford_median_sd(tb_smooth: pd.Series) -> tuple[float, float]:
    x = tb_smooth.dropna()
    x = x[(x > 28) & (x < 42)]
    if len(x) < MIN_BASE_POINTS:
        med = float(np.median(x.values)) if len(x) else 37.0
        sd = float(np.std(x.values, ddof=0)) if len(x) else 0.5
        return med, max(sd, 0.5)

    q_med = x.quantile(1.0 - MED_TOP_FRAC)
    eut = x[x >= q_med]
    med = float(np.median(eut.values)) if len(eut) else float(np.median(x.values))

    lo = x.quantile(SD_TRIM_LO)
    hi = x.quantile(SD_TRIM_HI)
    band = x[(x >= lo) & (x <= hi)]
    sd = float(np.std(band.values, ddof=0)) if len(band) else float(np.std(x.values, ddof=0))

    return med, max(sd, 0.3)


def add_oxford_ref_N(dfN: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (mid, src), sub in dfN.groupby(["mouse_id", "source_file"]):
        sub = sub.sort_values("datetime").reset_index(drop=True).copy()
        sub["Tb_smooth"] = rolling_time(sub, "Tb")

        med, sd = oxford_median_sd(sub["Tb_smooth"])
        sub["thr_gray_Tb"] = med - K_GRAY * sd
        sub["thr_deep_Tb"] = med - K_DEEP * sd

        ref_gray = sub["Tb_smooth"].values < sub["thr_gray_Tb"].iloc[0]
        ref_deep = sub["Tb_smooth"].values < sub["thr_deep_Tb"].iloc[0]

        sub["ref_gray"] = ref_gray
        sub["ref_deep_1h"] = keep_min_duration(ref_deep, dt_minutes(sub), MIN_DEEP_MIN)
        out.append(sub)

    return pd.concat(out, ignore_index=True) if out else dfN


# ---------------- Threshold calibration (ref quantiles + global floors) ----------------
def calibrate_thresholds(n: pd.DataFrame, q_deep_ref: float, q_gray_ref: float,
                         q_deep_global: float, q_gray_global: float):
    p_all = n["prob_s"].values
    deep_floor = float(np.quantile(p_all, q_deep_global))
    gray_floor = float(np.quantile(p_all, q_gray_global))

    deep_ref_vals, gray_ref_vals = [], []
    diag = []

    for mid, sub in n.groupby("mouse_id"):
        p = sub["prob_s"].values
        d = sub["ref_deep_1h"].values.astype(bool)
        g = sub["ref_gray"].values.astype(bool)
        if d.any():
            deep_ref_vals.append(p[d])
        if g.any():
            gray_ref_vals.append(p[g])
        diag.append(
            {
                "mouse_id": mid,
                "deep_ref_points": int(d.sum()),
                "gray_ref_points": int(g.sum()),
                "thr_deep_Tb": float(sub["thr_deep_Tb"].iloc[0]),
                "thr_gray_Tb": float(sub["thr_gray_Tb"].iloc[0]),
            }
        )

    deep_ref_all = np.concatenate(deep_ref_vals) if deep_ref_vals else np.array([])
    gray_ref_all = np.concatenate(gray_ref_vals) if gray_ref_vals else np.array([])

    deep_from_ref = float(np.quantile(deep_ref_all, q_deep_ref)) if deep_ref_all.size >= 50 else deep_floor
    gray_from_ref = float(np.quantile(gray_ref_all, q_gray_ref)) if gray_ref_all.size >= 50 else gray_floor

    # Final thresholds: must be at least the global floors (prevents degenerate low thresholds)
    deep_thr = max(deep_from_ref, deep_floor)
    gray_thr = max(gray_from_ref, gray_floor)

    # Enforce ordering
    if gray_thr >= deep_thr:
        gray_thr = deep_thr * 0.6

    return float(deep_thr), float(gray_thr), pd.DataFrame(diag), float(deep_floor), float(gray_floor)


# ---------------- States + duration constraints ----------------
def eventify(n: pd.DataFrame, deep_thr: float, gray_thr: float) -> pd.DataFrame:
    parts = []
    for mid, sub in n.groupby("mouse_id"):
        sub = sub.sort_values("datetime").reset_index(drop=True).copy()

        sub["state"] = np.where(
            sub["prob_s"] >= deep_thr,
            "Deep",
            np.where(sub["prob_s"] >= gray_thr, "Gray", "Normal"),
        )

        dt = dt_minutes(sub)

        deep_mask = sub["state"].values == "Deep"
        deep_keep = keep_min_duration(deep_mask, dt, MIN_DEEP_MIN)
        sub.loc[deep_mask & (~deep_keep), "state"] = "Gray"

        gray_mask = sub["state"].values == "Gray"
        gray_keep = keep_min_duration(gray_mask, dt, MIN_GRAY_MIN)
        sub.loc[gray_mask & (~gray_keep), "state"] = "Normal"

        parts.append(sub)

    return pd.concat(parts, ignore_index=True)


# ---------------- QC ----------------
def qc_metrics(n_out: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mid, sub in n_out.groupby("mouse_id"):
        sub = sub.sort_values("datetime").reset_index(drop=True)
        if sub.empty:
            continue

        t0 = sub["datetime"].iloc[0]
        first = sub[sub["datetime"] < (t0 + pd.Timedelta(hours=12))]
        denom = max(1, len(first))
        deep_frac_12h = float((first["state"] == "Deep").sum() / denom)
        gray_frac_12h = float((first["state"] == "Gray").sum() / denom)

        deep_state = sub["state"].values == "Deep"
        ref_deep = sub["ref_deep_1h"].values.astype(bool)
        deep_overlap_ratio = float((deep_state & ref_deep).sum() / max(1, deep_state.sum()))

        p = sub["prob_s"].values
        if ref_deep.any() and (~ref_deep).any():
            prob_sep = float(np.median(p[ref_deep]) - np.median(p[~ref_deep]))
        else:
            prob_sep = float("nan")

        dt = dt_minutes(sub)
        deep_min = float(deep_state.sum() * dt)
        gray_min = float((sub["state"].values == "Gray").sum() * dt)

        rows.append(
            {
                "mouse_id": mid,
                "deep_frac_12h": deep_frac_12h,
                "gray_frac_12h": gray_frac_12h,
                "deep_overlap_ratio": deep_overlap_ratio,
                "prob_sep": prob_sep,
                "deep_minutes": deep_min,
                "gray_minutes": gray_min,
            }
        )
    return pd.DataFrame(rows)


def min_bout_minutes(sub: pd.DataFrame, target: str) -> float:
    m = sub["state"].values == target
    sp = spans(m)
    if not sp:
        return 0.0
    dt = dt_minutes(sub)
    return float(min((b - a + 1) * dt for a, b in sp))


def qc_min_bouts(n_out: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mid, sub in n_out.groupby("mouse_id"):
        sub = sub.sort_values("datetime").reset_index(drop=True)
        rows.append(
            {
                "mouse_id": mid,
                "min_deep_bout_min": min_bout_minutes(sub, "Deep"),
                "min_gray_bout_min": min_bout_minutes(sub, "Gray"),
            }
        )
    return pd.DataFrame(rows)


# ---------------- Plotting ----------------
def plot_map(sub: pd.DataFrame, mid: str, deep_thr: float, gray_thr: float, out_png: Path):
    sub = sub.sort_values("datetime").reset_index(drop=True)
    if sub.empty:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 7.5), sharex=True)

    ax1.plot(sub["datetime"], sub["Tb"], color="black", lw=1.0, label="Tb")
    ax1.plot(sub["datetime"], sub["Tb_smooth"], color="gray", lw=1.0, alpha=0.75, label="Tb smooth (20min)")

    for a, b in spans(sub["state"].values == "Deep"):
        ax1.axvspan(sub["datetime"].iloc[a], sub["datetime"].iloc[b], alpha=0.25, color="red")
    for a, b in spans(sub["state"].values == "Gray"):
        ax1.axvspan(sub["datetime"].iloc[a], sub["datetime"].iloc[b], alpha=0.12, color="orange")

    ax1.axvline(FASTING_START, color="blue", linestyle="--", alpha=0.7, label="Fasting start")
    ax1.set_ylabel("Tb (°C)")
    ax1.set_title(f"Torpor Map: {mid} (drop first {DROP_FIRST_DAYS_N} days)")
    ax1.legend(loc="upper right")

    ax2.plot(sub["datetime"], sub["prob"], color="purple", lw=0.8, alpha=0.45, label="SVM prob (raw)")
    ax2.plot(sub["datetime"], sub["prob_s"], color="purple", lw=1.2, label="SVM prob (smooth)")
    ax2.axhline(deep_thr, color="red", linestyle=":", alpha=0.6, label=f"deep_thr={deep_thr:.4f}")
    ax2.axhline(gray_thr, color="orange", linestyle=":", alpha=0.6, label=f"gray_thr={gray_thr:.4f}")
    ax2.set_ylabel("Probability")
    ax2.set_xlabel("Time")
    ax2.set_ylim(0, 1)
    ax2.set_yticks([0, 0.5, 1])
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def select_best(sweep_df: pd.DataFrame) -> pd.Series:
    """
    No magic weights:
    - Guardrails: early 12h Deep small; early Gray not inflated (mean).
    - Objective: maximize prob_sep_mean (interpretable separation).
    - Tie-breaker: smaller gray_minutes_mean.
    """
    s = sweep_df.copy()
    s["prob_sep_mean"] = s["prob_sep_mean"].fillna(-1e9)

    ok = s[(s["deep_12h_max"] <= MAX_DEEP_FRAC_12H) & (s["gray_12h_mean"] <= MAX_GRAY_FRAC_12H_MEAN)].copy()
    if ok.empty:
        return s.sort_values(["deep_12h_max", "gray_12h_mean", "prob_sep_mean"], ascending=[True, True, False]).iloc[0]

    return ok.sort_values(["prob_sep_mean", "gray_minutes_mean"], ascending=[False, True]).iloc[0]


# ---------------- Main ----------------
def main():
    ensure(OUT)
    ensure(OUT / "plots")

    # Train SVM on F weak labels
    f = preprocess(load_F(F_DIR), is_N=False)
    f = weak_label_F(f)
    tr = f[f["label"] != -1].dropna(subset=FEATS).copy()
    if tr.empty or tr["label"].nunique() < 2:
        raise RuntimeError("F weak labels insufficient (<2 classes). Check baseline key and anchors.")

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", C=SVM_C, class_weight="balanced", probability=True)),
        ]
    )
    model.fit(tr[FEATS], tr["label"])
    print(f"[OK] Model=SVM. Train N={len(tr)} pos={(tr['label']==1).sum()} neg={(tr['label']==0).sum()}")

    # Load N, drop cold-start, build Oxford refs
    n = preprocess(load_N(N_DIR), is_N=True)
    if n.empty:
        raise RuntimeError("No N data loaded. Check N_DIR and column names.")
    n = add_oxford_ref_N(n)

    # Predict prob + 20min smoothing
    n = n.sort_values(["mouse_id", "source_file", "datetime"]).reset_index(drop=True)
    Xn = n[FEATS].fillna(0.0)
    n["prob"] = model.predict_proba(Xn)[:, 1]
    n["prob_s"] = n.groupby(["mouse_id", "source_file"], group_keys=False).apply(
        lambda s: rolling_time(s, "prob")
    ).reset_index(drop=True)

    # Sweep thresholds
    rows = []
    cache = {}
    for qd, qg, qdg, qgg in SWEEP:
        deep_thr, gray_thr, diag, deep_floor, gray_floor = calibrate_thresholds(n, qd, qg, qdg, qgg)
        n_out = eventify(n, deep_thr, gray_thr)
        qc = qc_metrics(n_out)

        row = {
            "q_deep_ref": qd, "q_gray_ref": qg, "q_deep_global": qdg, "q_gray_global": qgg,
            "deep_floor": deep_floor, "gray_floor": gray_floor,
            "deep_thr": deep_thr, "gray_thr": gray_thr,
            "deep_12h_max": float(qc["deep_frac_12h"].max()),
            "gray_12h_mean": float(qc["gray_frac_12h"].mean()),
            "prob_sep_mean": float(qc["prob_sep"].mean()),
            "deep_minutes_mean": float(qc["deep_minutes"].mean()),
            "gray_minutes_mean": float(qc["gray_minutes"].mean()),
        }
        rows.append(row)
        cache[(qd, qg, qdg, qgg)] = (n_out, diag, qc, qc_min_bouts(n_out))

    sweep = pd.DataFrame(rows).sort_values(["q_deep_global", "q_gray_global", "q_deep_ref", "q_gray_ref"]).reset_index(drop=True)
    sweep.to_csv(OUT / "q_sweep_summary.csv", index=False)

    best = select_best(sweep)
    key = (best["q_deep_ref"], best["q_gray_ref"], best["q_deep_global"], best["q_gray_global"])
    n_out, diag, qc, bout = cache[key]

    deep_thr = float(best["deep_thr"])
    gray_thr = float(best["gray_thr"])
    print(f"[OK] Selected thresholds: deep_thr={deep_thr:.4f}, gray_thr={gray_thr:.4f}")
    print(f"     floors: deep_floor={float(best['deep_floor']):.4f}, gray_floor={float(best['gray_floor']):.4f}")

    # Save outputs
    diag.to_csv(OUT / "oxford_calibration_diag.csv", index=False)
    qc.to_csv(OUT / "N_qc_per_mouse.csv", index=False)
    bout.to_csv(OUT / "N_min_bout_minutes.csv", index=False)
    qc[["mouse_id", "deep_frac_12h", "gray_frac_12h"]].to_csv(OUT / "N_sanity_first12h.csv", index=False)
    n_out.to_csv(OUT / "discovery_N16_oxford_drop2days.csv", index=False)

    # Plots: all N mice
    for mid in sorted(n_out["mouse_id"].unique()):
        sub = n_out[n_out["mouse_id"] == mid].copy()
        plot_map(sub, mid, deep_thr, gray_thr, OUT / "plots" / f"N16_{mid}_Map.png")

    print("[OK] Saved to:", OUT)


if __name__ == "__main__":
    main()
