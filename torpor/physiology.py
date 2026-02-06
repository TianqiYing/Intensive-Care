import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Load + preprocess + features
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

    # derivative invalid across time breaks
    df.loc[df["bad_time"], ["dT", "dTdt_C_per_min"]] = np.nan
    return df


def truncate_after_first_rollover(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index[df["rollover_flag"].fillna(False)]
    if len(idx) == 0:
        return df
    return df.iloc[: int(idx[0])].reset_index(drop=True)


def add_rolling_features(df: pd.DataFrame, win_min: int = 60, min_points: int = 12) -> pd.DataFrame:
    """
    roc_smooth = robust rolling median of dT/dt (direction signal)
    roc_sd     = rolling SD of dT/dt (variability signal)
    """
    df = df.copy()
    roc = df["dTdt_C_per_min"].replace([np.inf, -np.inf], np.nan)

    dt_med = df["dt_min"].dropna()
    dt_med = dt_med[(dt_med > 0) & (dt_med < 30)]
    if len(dt_med) == 0:
        # cannot estimate timestep -> skip
        df["roc_smooth"] = np.nan
        df["roc_sd"] = np.nan
        return df

    step = float(dt_med.median())
    win = max(3, int(round(win_min / step)))

    df["roc_smooth"] = roc.rolling(win, min_periods=min_points, center=True).median()
    df["roc_sd"] = roc.rolling(win, min_periods=min_points, center=True).std()
    return df


# ----------------------------
# Step 1: Symbolic dynamics (quantile states)
# ----------------------------

def symbolise_roc(df: pd.DataFrame, q_low=0.2, q_high=0.8) -> tuple[pd.DataFrame, dict]:
    """
    Convert roc_smooth into symbols using per-mouse quantiles.
      C: roc_smooth <= q_low quantile
      W: roc_smooth >= q_high quantile
      N: otherwise
    """
    df = df.copy()
    x = df["roc_smooth"].dropna()
    if len(x) < 50:
        df["state"] = "N"
        return df, {"q_low": np.nan, "q_high": np.nan}

    lo = float(x.quantile(q_low))
    hi = float(x.quantile(q_high))

    df["state"] = "N"
    df.loc[df["roc_smooth"] <= lo, "state"] = "C"
    df.loc[df["roc_smooth"] >= hi, "state"] = "W"

    return df, {"q_low": lo, "q_high": hi}


# ----------------------------
# Step 2: Find C-run then W-run (unlabelled “event”)
# ----------------------------

def _run_length_minutes(times: pd.Series) -> float:
    """Duration in minutes from first to last timestamp in a run."""
    if len(times) < 2:
        return 0.0
    return float((times.iloc[-1] - times.iloc[0]).total_seconds() / 60.0)


def find_cw_events(
    df: pd.DataFrame,
    min_run_min: float = 120.0,     # minimum duration for C or W run (minutes)
    max_gap_between_min: float = 360.0,  # max allowed time between end of C and start of W
    search_after_c_min: float = 1440.0,  # search for W within this time after C ends
) -> list[dict]:
    """
    Scan symbolic sequence for cooling run (C) followed by warming run (W).
    No temperature thresholds; purely dynamics + persistence.
    """
    d = df.dropna(subset=["Time"]).copy()
    if "state" not in d.columns:
        return []

    # Identify contiguous runs by state
    # A new run starts whenever state changes OR there's a bad_time step
    run_break = (d["state"] != d["state"].shift(1)) | (d.get("bad_time", False).fillna(False))
    d["run_id"] = run_break.cumsum()

    runs = (
        d.groupby(["run_id", "state"], as_index=False)
         .agg(start=("Time", "first"),
              end=("Time", "last"),
              n=("Time", "size"))
    )

    # Compute duration (min) for each run
    runs["dur_min"] = (runs["end"] - runs["start"]).dt.total_seconds() / 60.0
    events = []

    # index runs in time order
    runs = runs.sort_values("start").reset_index(drop=True)

    for i in range(len(runs)):
        r = runs.iloc[i]
        if r["state"] != "C":
            continue
        if r["dur_min"] < min_run_min:
            continue

        c_start, c_end = r["start"], r["end"]

        # Find the first W-run that starts after this C-run, within constraints
        # (allow a small neutral gap but not huge)
        w_candidates = runs[
            (runs["state"] == "W") &
            (runs["start"] >= c_end) &
            (runs["start"] <= c_end + pd.Timedelta(minutes=search_after_c_min))
        ]

        if len(w_candidates) == 0:
            continue

        w = w_candidates.iloc[0]
        gap_min = (w["start"] - c_end).total_seconds() / 60.0
        if gap_min > max_gap_between_min:
            continue
        if w["dur_min"] < min_run_min:
            continue

        w_start, w_end = w["start"], w["end"]

        # Event window: from C start to W end
        ev_start, ev_end = c_start, w_end
        ev = d[(d["Time"] >= ev_start) & (d["Time"] <= ev_end)].copy()

        # Summarise dynamics within event
        cool_peak = float(ev["roc_smooth"].min()) if ev["roc_smooth"].notna().any() else np.nan
        warm_peak = float(ev["roc_smooth"].max()) if ev["roc_smooth"].notna().any() else np.nan
        tb_min = float(ev["Temperature"].min()) if ev["Temperature"].notna().any() else np.nan

        # variability signature
        roc_sd_mean = float(ev["roc_sd"].mean()) if ev["roc_sd"].notna().any() else np.nan

        events.append(dict(
            event_start=ev_start,
            event_end=ev_end,
            cooling_start=c_start,
            cooling_end=c_end,
            warming_start=w_start,
            warming_end=w_end,
            cooling_dur_min=float(r["dur_min"]),
            warming_dur_min=float(w["dur_min"]),
            gap_C_to_W_min=float(gap_min),
            roc_smooth_min=cool_peak,
            roc_smooth_max=warm_peak,
            temp_min=tb_min,
            roc_sd_mean=roc_sd_mean,
        ))

    return events


# ----------------------------
# Step 3: Phase portrait plots for each event
# ----------------------------

def plot_phase_portrait(ev_df: pd.DataFrame, title: str, outpath: str):
    """
    Phase portrait: Temperature vs roc_smooth during the event.
    Uses time as the scatter color variable (default matplotlib colormap).
    """
    x = ev_df["Temperature"].to_numpy(dtype=float)
    y = ev_df["roc_smooth"].to_numpy(dtype=float)

    # remove NaNs
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 20:
        return

    t = ev_df.loc[mask, "Time"].astype("int64")  # numeric for coloring
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x[mask], y[mask], c=t)  # default colormap
    ax.set_title(title)
    ax.set_xlabel("Temperature")
    ax.set_ylabel("roc_smooth (°C/min)")
    plt.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


# ----------------------------
# Main runner: detect events + export
# ----------------------------

def run_symbolic_event_detection(
    pattern="N*.xlsx",
    out_excel="torpor_symbolic_events.xlsx",
    plots_dir="plots_symbolic_events",
    win_min=60,
    min_points=12,
    q_low=0.2,
    q_high=0.8,
    min_run_min=120.0,
    max_gap_between_min=360.0
):
    os.makedirs(plots_dir, exist_ok=True)

    all_events = []
    symbol_params = []
    per_mouse_counts = []

    for path in sorted(glob.glob(pattern)):
        mouse = os.path.splitext(os.path.basename(path))[0]

        df = load_one_file(path)
        df = preprocess_basic(df, drop_first_days=3, max_temp=45.0)
        df = flag_time_problems(df, max_gap_min=30.0)
        df = truncate_after_first_rollover(df)
        df = add_rolling_features(df, win_min=win_min, min_points=min_points)

        df, q = symbolise_roc(df, q_low=q_low, q_high=q_high)
        symbol_params.append({"mouse": mouse, **q, "q_low_prob": q_low, "q_high_prob": q_high})

        events = find_cw_events(
            df,
            min_run_min=min_run_min,
            max_gap_between_min=max_gap_between_min,
            search_after_c_min=1440.0
        )

        per_mouse_counts.append({"mouse": mouse, "n_events": len(events)})

        # save phase portraits per event + collect results
        for j, ev in enumerate(events, start=1):
            ev_df = df[(df["Time"] >= ev["event_start"]) & (df["Time"] <= ev["event_end"])].copy()
            title = f"{mouse} event{j} | {ev['event_start']} → {ev['event_end']}"
            outpath = os.path.join(plots_dir, f"{mouse}_event{j}_phase.png")
            plot_phase_portrait(ev_df, title=title, outpath=outpath)

            ev_row = {"mouse": mouse, "event_id": j, **ev}
            all_events.append(ev_row)

    events_df = pd.DataFrame(all_events).sort_values(["mouse", "event_start"]) if len(all_events) else pd.DataFrame()
    symbol_df = pd.DataFrame(symbol_params).sort_values("mouse")
    counts_df = pd.DataFrame(per_mouse_counts).sort_values("mouse")

    with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
        symbol_df.to_excel(writer, sheet_name="symbol_params", index=False)
        counts_df.to_excel(writer, sheet_name="per_mouse_counts", index=False)
        events_df.to_excel(writer, sheet_name="events", index=False)

    print(f"\nWrote: {out_excel}")
    print(f"Saved phase portraits to: {plots_dir}/")
    print("\nInterpretation reminder:")
    print("- Events are detected as sustained C-run followed by sustained W-run, where C/W are defined by per-mouse quantiles of roc_smooth.")
    print("- No temperature thresholds were used. This is unlabelled exploration of structured dynamics.")


if __name__ == "__main__":
    run_symbolic_event_detection(
        pattern="N*.xlsx",
        out_excel="torpor_symbolic_events.xlsx",
        plots_dir="plots_symbolic_events",
        win_min=60,
        min_points=12,
        q_low=0.2,
        q_high=0.8,
        min_run_min=120.0,
        max_gap_between_min=360.0
    )
