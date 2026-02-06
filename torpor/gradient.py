import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Core pipeline
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

    # Drop first N days
    cutoff = df["Time"].iloc[0] + pd.Timedelta(days=drop_first_days)
    df = df[df["Time"] >= cutoff].reset_index(drop=True)

    # Only remove extreme highs (keep low temps)
    df.loc[df["Temperature"] > max_temp, "Temperature"] = np.nan

    # dt and dT/dt (time-aware: uses actual dt between adjacent samples)
    df["dt_min"] = df["Time"].diff().dt.total_seconds() / 60.0
    df["dT"] = df["Temperature"].diff()
    df["dTdt_C_per_min"] = df["dT"] / df["dt_min"]

    return df


def flag_time_problems(df: pd.DataFrame, max_gap_min: float = 30.0) -> pd.DataFrame:
    df = df.copy()
    df["gap_flag"] = df["dt_min"] > max_gap_min
    df["time_backwards"] = df["dt_min"] < 0
    df["rollover_flag"] = df["dt_min"] > 1e5  # huge jump -> reset/rollover
    df["bad_time"] = df[["gap_flag", "time_backwards", "rollover_flag"]].any(axis=1)

    # Derivatives across these are invalid
    df.loc[df["bad_time"], ["dT", "dTdt_C_per_min"]] = np.nan
    return df


def truncate_after_first_rollover(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index[df["rollover_flag"].fillna(False)]
    if len(idx) == 0:
        return df
    return df.iloc[: int(idx[0])].reset_index(drop=True)


def add_rolling_features(df: pd.DataFrame, win_min: int = 60, min_points: int = 12) -> pd.DataFrame:
    """
    Computes:
      roc_smooth = rolling median of dT/dt (robust smoothed gradient)
      roc_sd     = rolling SD of dT/dt (variability of gradient)  <-- MOST IMPORTANT
    """
    df = df.copy()
    roc = df["dTdt_C_per_min"].replace([np.inf, -np.inf], np.nan)

    dt_med = df["dt_min"].dropna()
    dt_med = dt_med[(dt_med > 0) & (dt_med < 30)]
    step = float(dt_med.median())  # median timestep in minutes
    win = max(3, int(round(win_min / step)))

    df["roc_smooth"] = roc.rolling(win, min_periods=min_points, center=True).median()
    df["roc_sd"] = roc.rolling(win, min_periods=min_points, center=True).std()

    return df


def changepoint_score(series: pd.Series, win: int, min_points: int) -> pd.Series:
    x = series.replace([np.inf, -np.inf], np.nan).astype(float)

    mean_before = x.rolling(win, min_periods=min_points).mean()
    mean_after = x[::-1].rolling(win, min_periods=min_points).mean()[::-1]

    sd_before = x.rolling(win, min_periods=min_points).std()
    sd_after = x[::-1].rolling(win, min_periods=min_points).std()[::-1]

    pooled = np.sqrt((sd_before**2 + sd_after**2) / 2)
    return (mean_before - mean_after).abs() / pooled


def add_changepoint_scores(df: pd.DataFrame, score_win_min: int = 120) -> pd.DataFrame:
    df = df.copy()

    dt_med = df["dt_min"].dropna()
    dt_med = dt_med[(dt_med > 0) & (dt_med < 30)]
    step = float(dt_med.median())
    win = max(10, int(round(score_win_min / step)))
    min_points = max(10, win // 3)

    # Score changes in roc_smooth and roc_sd, then combine
    df["cp_score_roc"] = changepoint_score(df["roc_smooth"], win=win, min_points=min_points)
    df["cp_score_sd"] = changepoint_score(df["roc_sd"], win=win, min_points=min_points)
    df["cp_score_combined"] = df["cp_score_roc"].fillna(0) + df["cp_score_sd"].fillna(0)

    return df


# ----------------------------
# Switch time & slicing
# ----------------------------

def get_switch_time(df: pd.DataFrame,
                    window_start="2023-09-05 00:00:00",
                    window_end="2023-09-07 23:59:59") -> pd.Timestamp:
    ws = pd.to_datetime(window_start)
    we = pd.to_datetime(window_end)
    d = df.dropna(subset=["cp_score_combined"]).copy()
    w = d[(d["Time"] >= ws) & (d["Time"] <= we)]
    if len(w) == 0:
        return pd.NaT
    return w.loc[w["cp_score_combined"].idxmax(), "Time"]


def slice_window(df: pd.DataFrame, center: pd.Timestamp, hours: int = 48) -> pd.DataFrame:
    if pd.isna(center):
        return df.iloc[0:0].copy()
    start = center - pd.Timedelta(hours=hours)
    end = center + pd.Timedelta(hours=hours)
    return df[(df["Time"] >= start) & (df["Time"] <= end)].copy()


# ----------------------------
# Plotting (separate, uncluttered)
# ----------------------------

def plot_single_series(time, y, title, ylabel, outpath, vline_time=None, hline0=False):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time, y)
    if vline_time is not None and not pd.isna(vline_time):
        ax.axvline(vline_time, linestyle="--")
    if hline0:
        ax.axhline(0, linestyle=":", alpha=0.5)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def make_plots_for_mouse(df: pd.DataFrame, name: str, t0: pd.Timestamp, plots_dir: str, hours: int = 48):
    os.makedirs(plots_dir, exist_ok=True)

    w = slice_window(df, t0, hours=hours)
    if len(w) == 0:
        return

    # 1) Temperature
    plot_single_series(
        w["Time"], w["Temperature"],
        title=f"{name} | Temperature ±{hours}h around switch {t0}",
        ylabel="Temperature",
        outpath=os.path.join(plots_dir, f"{name}_temp_switch.png"),
        vline_time=t0
    )

    # 2) Smoothed gradient (direction)
    plot_single_series(
        w["Time"], w["roc_smooth"],
        title=f"{name} | Smoothed dT/dt ±{hours}h around switch {t0}",
        ylabel="Smoothed dT/dt (°C/min)",
        outpath=os.path.join(plots_dir, f"{name}_roc_smooth_switch.png"),
        vline_time=t0,
        hline0=True
    )

    # 3) Variability of gradient (THIS is the most important)
    plot_single_series(
        w["Time"], w["roc_sd"],
        title=f"{name} | Variability of gradient (SD of dT/dt) ±{hours}h around switch {t0}",
        ylabel="SD(dT/dt)",
        outpath=os.path.join(plots_dir, f"{name}_roc_sd_switch.png"),
        vline_time=t0
    )


# ----------------------------
# Export summaries to Excel
# ----------------------------

def dt_summary(df: pd.DataFrame) -> dict:
    dt = df["dt_min"].dropna()
    dt = dt[(dt > 0) & (dt < 1e5)]
    if len(dt) == 0:
        return dict(median_dt_min=np.nan, p95_dt_min=np.nan, max_dt_min=np.nan)
    return dict(
        median_dt_min=float(dt.median()),
        p95_dt_min=float(dt.quantile(0.95)),
        max_dt_min=float(dt.max()),
    )


def build_and_export(pattern="N*.xlsx",
                     out_excel="torpor_exploration_results.xlsx",
                     plots_dir="plots_clean",
                     hours=48):
    files = sorted(glob.glob(pattern))
    data = {}

    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        df = load_one_file(path)
        df = preprocess_basic(df, drop_first_days=3, max_temp=45.0)
        df = flag_time_problems(df, max_gap_min=30.0)
        df = truncate_after_first_rollover(df)
        df = add_rolling_features(df, win_min=60, min_points=12)
        df = add_changepoint_scores(df, score_win_min=120)
        data[name] = df

    switch_rows = []
    dt_rows = []
    excerpt_rows = []

    for name, df in data.items():
        t0 = get_switch_time(df)

        # switch score (if present)
        switch_score = np.nan
        if not pd.isna(t0):
            switch_score = float(df.loc[df["Time"].eq(t0), "cp_score_combined"].iloc[0])

        switch_rows.append(dict(mouse=name, switch_time=t0, switch_cp_score=switch_score))
        dt_rows.append(dict(mouse=name, **dt_summary(df)))

        # plots (3 separate plots per mouse)
        make_plots_for_mouse(df, name, t0, plots_dir=plots_dir, hours=hours)

        # export a tidy excerpt around switch for downstream analysis
        if not pd.isna(t0):
            w = slice_window(df, t0, hours=hours)
            w = w[["Time", "Temperature", "dt_min", "dTdt_C_per_min", "roc_smooth", "roc_sd", "cp_score_combined"]].copy()
            w.insert(0, "mouse", name)
            w.insert(1, "t_rel_min", (w["Time"] - t0).dt.total_seconds() / 60.0)
            excerpt_rows.append(w)

    switch_df = pd.DataFrame(switch_rows).sort_values("mouse")
    dt_df = pd.DataFrame(dt_rows).sort_values("mouse")
    excerpt_df = pd.concat(excerpt_rows, ignore_index=True) if len(excerpt_rows) else pd.DataFrame()

    with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
        switch_df.to_excel(writer, sheet_name="switch_summary", index=False)
        dt_df.to_excel(writer, sheet_name="dt_summary", index=False)
        if len(excerpt_df):
            excerpt_df.to_excel(writer, sheet_name=f"switch_excerpt_{hours}h", index=False)

    print(f"\nWrote Excel: {out_excel}")
    print(f"Saved plots to: {plots_dir}/")
    print("\nImportant note about dT/dt timestep:")
    print("dT/dt is computed between adjacent samples using the actual dt:")
    print("dTdt_C_per_min[i] = (T[i] - T[i-1]) / dt_min[i]  where dt_min is in minutes.")
    print("So the 'timestep' varies (usually ~1–5 min, sometimes larger), and we set dT/dt=NaN across big gaps.")


if __name__ == "__main__":
    build_and_export(
        pattern="N*.xlsx",
        out_excel="torpor_exploration_results.xlsx",
        plots_dir="plots_clean",
        hours=48
    )
