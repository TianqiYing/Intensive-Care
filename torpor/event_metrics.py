import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Rebuild the same time-series features (so event metrics are consistent)
# ----------------------------

def load_one_file(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = df[["Time", "Temperature"]].copy()
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df["Temperature"] = pd.to_numeric(df["Temperature"], errors="coerce")
    df = df.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)
    return df


def preprocess_basic(df: pd.DataFrame, drop_first_days: int = 3, max_temp: float = 45.0) -> pd.DataFrame:
    df = df.copy()
    cutoff = df["Time"].iloc[0] + pd.Timedelta(days=drop_first_days)
    df = df[df["Time"] >= cutoff].reset_index(drop=True)

    # keep low temps; remove extreme highs only
    df.loc[df["Temperature"] > max_temp, "Temperature"] = np.nan

    # dT/dt uses *actual dt* between adjacent samples (variable timestep)
    df["dt_min"] = df["Time"].diff().dt.total_seconds() / 60.0
    df["dT"] = df["Temperature"].diff()
    df["dTdt_C_per_min"] = df["dT"] / df["dt_min"]
    return df


def flag_time_problems(df: pd.DataFrame, max_gap_min: float = 30.0) -> pd.DataFrame:
    df = df.copy()
    df["gap_flag"] = df["dt_min"] > max_gap_min
    df["time_backwards"] = df["dt_min"] < 0
    df["rollover_flag"] = df["dt_min"] > 1e5
    df["bad_time"] = df[["gap_flag", "time_backwards", "rollover_flag"]].any(axis=1)

    # invalidate derivatives across broken time steps
    df.loc[df["bad_time"], ["dT", "dTdt_C_per_min"]] = np.nan
    return df


def truncate_after_first_rollover(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index[df["rollover_flag"].fillna(False)]
    if len(idx) == 0:
        return df
    return df.iloc[: int(idx[0])].reset_index(drop=True)


def add_rolling_features(df: pd.DataFrame, win_min: int = 60, min_points: int = 12) -> pd.DataFrame:
    df = df.copy()
    roc = df["dTdt_C_per_min"].replace([np.inf, -np.inf], np.nan)

    dt_med = df["dt_min"].dropna()
    dt_med = dt_med[(dt_med > 0) & (dt_med < 30)]
    if len(dt_med) == 0:
        df["roc_smooth"] = np.nan
        df["roc_sd"] = np.nan
        return df

    step = float(dt_med.median())
    win = max(3, int(round(win_min / step)))

    # roc_smooth: direction; roc_sd: variability of gradient
    df["roc_smooth"] = roc.rolling(win, min_periods=min_points, center=True).median()
    df["roc_sd"] = roc.rolling(win, min_periods=min_points, center=True).std()
    return df


def build_mouse_df(mouse: str) -> pd.DataFrame:
    df = load_one_file(f"{mouse}.xlsx")
    df = preprocess_basic(df, drop_first_days=3, max_temp=45.0)
    df = flag_time_problems(df, max_gap_min=30.0)
    df = truncate_after_first_rollover(df)
    df = add_rolling_features(df, win_min=60, min_points=12)
    return df


# ----------------------------
# Event metrics (the “proper indicator” ingredients)
# ----------------------------

def phase_ellipse_area(temp: np.ndarray, roc: np.ndarray) -> float:
    """
    Estimate phase-space area via 1-sigma covariance ellipse:
      A = pi * sqrt(lambda1) * sqrt(lambda2)
    where lambdas are eigenvalues of covariance of [temp, roc].
    """
    x = np.asarray(temp, float)
    y = np.asarray(roc, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if len(x) < 20:
        return np.nan

    X = np.column_stack([x, y])
    cov = np.cov(X, rowvar=False)
    if not np.all(np.isfinite(cov)):
        return np.nan

    vals = np.linalg.eigvalsh(cov)  # eigenvalues
    vals = np.maximum(vals, 0.0)
    return float(np.pi * np.sqrt(vals[0]) * np.sqrt(vals[1]))


def directional_asymmetry(roc: np.ndarray) -> tuple[float, float, float]:
    """
    Split event into first half and second half (by time order).
    Return mean1, mean2, delta = mean2 - mean1.
    """
    r = np.asarray(roc, float)
    r = r[np.isfinite(r)]
    if len(r) < 20:
        return (np.nan, np.nan, np.nan)

    mid = len(r) // 2
    a = r[:mid]
    b = r[mid:]
    if len(a) < 10 or len(b) < 10:
        return (np.nan, np.nan, np.nan)

    m1 = float(np.mean(a))
    m2 = float(np.mean(b))
    return (m1, m2, float(m2 - m1))


def compute_event_metrics(mouse_df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    ev = mouse_df[(mouse_df["Time"] >= start) & (mouse_df["Time"] <= end)].copy()
    if len(ev) < 20:
        return {}

    temp = ev["Temperature"].to_numpy()
    roc = ev["roc_smooth"].to_numpy()
    roc_sd = ev["roc_sd"].to_numpy()

    area = phase_ellipse_area(temp, roc)
    m1, m2, d = directional_asymmetry(roc)

    return dict(
        n_points=int(len(ev)),
        duration_min=float((end - start).total_seconds() / 60.0),
        phase_area=area,
        roc_mean_first_half=m1,
        roc_mean_second_half=m2,
        roc_asymmetry_delta=d,
        roc_sd_mean=float(np.nanmean(roc_sd)) if np.isfinite(roc_sd).any() else np.nan,
        temp_min=float(np.nanmin(temp)) if np.isfinite(temp).any() else np.nan,
        temp_max=float(np.nanmax(temp)) if np.isfinite(temp).any() else np.nan,
    )


# ----------------------------
# Main: read your events Excel, add metrics, export + plots
# ----------------------------

def main(events_excel="torpor_symbolic_events.xlsx",
         out_excel="torpor_symbolic_events_with_metrics.xlsx",
         plots_dir="plots_event_metrics"):

    os.makedirs(plots_dir, exist_ok=True)

    xls = pd.read_excel(events_excel, sheet_name=None)
    if "events" not in xls:
        raise ValueError("Could not find an 'events' sheet in torpor_symbolic_events.xlsx")

    events = xls["events"].copy()
    if len(events) == 0:
        print("No events found in the Excel file.")
        return

    # Ensure datetime
    for col in ["event_start", "event_end"]:
        events[col] = pd.to_datetime(events[col])

    # Build per-mouse cached dataframes
    mouse_dfs = {}
    rows = []

    for i, row in events.iterrows():
        mouse = row["mouse"]
        if mouse not in mouse_dfs:
            mouse_dfs[mouse] = build_mouse_df(mouse)

        start = row["event_start"]
        end = row["event_end"]

        m = compute_event_metrics(mouse_dfs[mouse], start, end)
        m.update({"mouse": mouse, "event_id": row.get("event_id", np.nan),
                  "event_start": start, "event_end": end})
        rows.append(m)

    metrics = pd.DataFrame(rows)

    # Merge metrics back into events table
    key_cols = ["mouse", "event_start", "event_end"]
    out_events = events.merge(metrics, on=key_cols, how="left", suffixes=("", "_m"))

    # Write new Excel (keep original sheets + add metrics)
    with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
        for name, df in xls.items():
            df.to_excel(writer, sheet_name=name, index=False)
        out_events.to_excel(writer, sheet_name="events_with_metrics", index=False)
        metrics.to_excel(writer, sheet_name="event_metrics_only", index=False)

    print(f"\nWrote: {out_excel}")
    print("Key new columns:")
    print("- phase_area: size of excursion in (Temperature, roc_smooth) space")
    print("- roc_asymmetry_delta: (mean roc second half) - (mean roc first half)")
    print("- roc_sd_mean: mean variability of gradient during event")

    # ---- Plots: let the “torpor-like” cluster reveal itself ----
    # 1) phase_area vs asymmetry
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(metrics["phase_area"], metrics["roc_asymmetry_delta"])
    ax.set_xlabel("phase_area (cov ellipse)")
    ax.set_ylabel("roc_asymmetry_delta (2nd half - 1st half)")
    ax.set_title("Events in feature space (should separate naturally)")
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "events_phasearea_vs_asymmetry.png"), dpi=200)
    plt.close(fig)

    # 2) phase_area vs roc_sd_mean (variability)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(metrics["phase_area"], metrics["roc_sd_mean"])
    ax.set_xlabel("phase_area (cov ellipse)")
    ax.set_ylabel("roc_sd_mean (variability of gradient)")
    ax.set_title("Events: excursion size vs gradient variability")
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "events_phasearea_vs_rocsd.png"), dpi=200)
    plt.close(fig)

    print(f"Saved plots to: {plots_dir}/")


if __name__ == "__main__":
    main(
        events_excel="torpor_symbolic_events.xlsx",
        out_excel="torpor_symbolic_events_with_metrics.xlsx",
        plots_dir="plots_event_metrics"
    )
