from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FAST_START = pd.Timestamp("2023-09-05 12:00")


def load_mouse_N(mouse_name: str, root: Path) -> pd.DataFrame:
    path = root / f"{mouse_name}.xlsx"
    df = pd.read_excel(path, engine="openpyxl")
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df = df.dropna(subset=["Time", "Temperature"]).sort_values("Time")
    return df


def build_empirical_baseline(df_base: pd.DataFrame) -> pd.Series:
    tod_hours = (
        df_base["Time"].dt.hour
        + df_base["Time"].dt.minute / 60
        + df_base["Time"].dt.second / 3600
    )
    df_tmp = df_base.copy()
    df_tmp["tod"] = tod_hours
    df_tmp["hour"] = df_tmp["tod"].astype(int)  # 0..23

    hourly_mean = df_tmp.groupby("hour")["Temperature"].mean()
    hourly_mean = hourly_mean.reindex(range(24)).interpolate().bfill().ffill()
    return hourly_mean  # index 0..23


def eval_baseline_from_hourly(hour_float, hourly_mean: pd.Series):
    h = np.asarray(hour_float)
    h_floor = np.floor(h).astype(int) % 24
    h_ceil = (h_floor + 1) % 24
    frac = h - np.floor(h)
    vals = hourly_mean.values
    T0 = vals[h_floor]
    T1 = vals[h_ceil]
    return (1.0 - frac) * T0 + frac * T1


def compute_residual_post_fast(
    mouse_name: str,
    project_root: Path,
    window_hours: float = 24.0,
):
    df = load_mouse_N(mouse_name, project_root)

    df_base = df[df["Time"] < FAST_START]
    df_post = df[
        (df["Time"] >= FAST_START)
        & (df["Time"] <= FAST_START + pd.Timedelta(hours=window_hours))
    ]
    if df_base.empty or df_post.empty:
        raise ValueError(f"{mouse_name}: baseline or post-fast window is empty.")

    hourly_mean = build_empirical_baseline(df_base)

    tod_post = (
        df_post["Time"].dt.hour
        + df_post["Time"].dt.minute / 60
        + df_post["Time"].dt.second / 3600
    )
    Tobs = df_post["Temperature"].astype(float).values
    Tbase = eval_baseline_from_hourly(tod_post.values, hourly_mean)
    resid_raw = Tobs - Tbase

    # resample to regular 10-min intervals
    ts = pd.Series(resid_raw, index=df_post["Time"])
    ts = ts.resample("10min").mean().interpolate("time")

    resid = ts.values
    t_hours = (ts.index - ts.index[0]).total_seconds() / 3600.0

    return ts.index, t_hours, resid
    # return both timestamps and hour-since-fast-start


def classify_torpor_phases(
    t_hours,
    resid,
    thr_entry=-1.5,
    thr_deep=-6.0,
    thr_exit=-1.5,
    smooth_win=5,
):
    r = resid.copy()

    # Smooth residual with moving average to reduce noise
    if smooth_win > 1:
        kernel = np.ones(smooth_win) / smooth_win
        r_smooth = np.convolve(r, kernel, mode="same")
    else:
        r_smooth = r

    # Approximate d(resid)/dt (°C per hour)
    dt = np.median(np.diff(t_hours))
    drdt = np.gradient(r_smooth, dt)

    n = len(r_smooth)

    idx_entry_candidates = np.where(r_smooth < thr_entry)[0]
    if len(idx_entry_candidates) == 0:
        idx_entry = 0
    else:
        idx_entry = idx_entry_candidates[0]

    idx_deep_candidates = np.where(r_smooth < thr_deep)[0]
    if len(idx_deep_candidates) == 0:
        idx_deep_start = idx_entry
    else:
        idx_deep_start = idx_deep_candidates[0]

    idx_min = np.argmin(r_smooth)

    idx_rewarm_candidates = np.where(
        (np.arange(n) > idx_min) & (drdt > 0.3)
    )[0]
    if len(idx_rewarm_candidates) == 0:
        idx_rewarm_start = idx_min
    else:
        idx_rewarm_start = idx_rewarm_candidates[0]

    idx_recovered_candidates = np.where(
        (np.arange(n) > idx_rewarm_start) & (r_smooth > thr_exit)
    )[0]
    if len(idx_recovered_candidates) == 0:
        idx_recovered = n - 1
    else:
        idx_recovered = idx_recovered_candidates[0]

    # enforce monotonic order
    idx_entry = max(0, min(idx_entry, n - 1))
    idx_deep_start = max(idx_entry, min(idx_deep_start, n - 1))
    idx_min = max(idx_deep_start, min(idx_min, n - 1))
    idx_rewarm_start = max(idx_min, min(idx_rewarm_start, n - 1))
    idx_recovered = max(idx_rewarm_start, min(idx_recovered, n - 1))

    # assign phase labels
    labels = np.zeros(n, dtype=int)
    labels[:idx_entry] = 0                       # baseline_before
    labels[idx_entry:idx_deep_start] = 1         # entering_torpor
    labels[idx_deep_start:idx_rewarm_start] = 2  # deep_torpor_plateau
    labels[idx_rewarm_start:idx_recovered] = 3   # exiting_torpor
    labels[idx_recovered:] = 4                   # baseline_after

    cut_times = dict(
        t_entry=t_hours[idx_entry],
        t_deep_start=t_hours[idx_deep_start],
        t_min=t_hours[idx_min],
        t_rewarm_start=t_hours[idx_rewarm_start],
        t_recovered=t_hours[idx_recovered],
    )

    return labels, cut_times


PHASE_NAMES = {
    0: "baseline_before",
    1: "entering_torpor",
    2: "deep_torpor_plateau",
    3: "exiting_torpor",
    4: "baseline_after",
}

PHASE_COLORS = {
    0: "tab:green",
    1: "tab:orange",
    2: "tab:red",
    3: "tab:purple",
    4: "tab:blue",
}


def plot_phases(mouse_name, t_hours, resid, labels, cut_times):
    plt.figure(figsize=(9, 3))
    for ph in range(5):
        mask = labels == ph
        plt.scatter(
            t_hours[mask],
            resid[mask],
            s=14,
            label=f"{ph}: {PHASE_NAMES[ph]}",
            color=PHASE_COLORS[ph],
        )
    plt.axhline(0, linestyle="--", color="gray")

    # vertical lines at phase boundaries
    for key, t in cut_times.items():
        plt.axvline(t, linestyle=":", color="gray", alpha=0.5)
        plt.text(
            t,
            np.min(resid) - 0.5,
            key,
            rotation=90,
            va="bottom",
            ha="right",
            fontsize=8,
        )

    plt.xlabel("Hours since fast start")
    plt.ylabel("Residual temperature (°C)")
    plt.title(f"{mouse_name}: 5 physiological phases")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.show()


def summarize_phase_durations(mouse_name, t_hours, labels):
    """
    Print approximate duration (in hours) of each phase for a mouse.
    """
    dt = np.median(np.diff(t_hours))
    print(f"\n=== {mouse_name}: phase durations (approx.) ===")
    for ph in range(5):
        n_points = np.sum(labels == ph)
        duration_h = n_points * dt
        print(f"  Phase {ph} ({PHASE_NAMES[ph]}): {duration_h:.2f} h")


def main():
    root = Path(__file__).resolve().parent
    mice = ["N9", "N10", "N12", "N14", "N15", "N16"]

    all_summaries = []

    for mouse_name in mice:
        print(f"\n\n----- Processing {mouse_name} -----")
        time_index, t_hours, resid = compute_residual_post_fast(
            mouse_name, root, window_hours=24.0
        )

        labels, cut_times = classify_torpor_phases(
            t_hours,
            resid,
            thr_entry=-1.5,   # same thresholds for all mice
            thr_deep=-6.0,
            thr_exit=-1.5,
            smooth_win=5,
        )

        print("Phase boundaries (hours since fast start):")
        for k, v in cut_times.items():
            print(f"  {k}: {v:.2f} h")

        summarize_phase_durations(mouse_name, t_hours, labels)

        # Save per-timepoint labels to CSV
        df_out = pd.DataFrame(
            {
                "Time": time_index,
                "hours_since_fast_start": t_hours,
                "residual_temp": resid,
                "phase": labels,
            }
        )
        out_path = root / f"torpor_phases_{mouse_name}.csv"
        df_out.to_csv(out_path, index=False)
        print(f"Saved phase labels to {out_path.name}")

        # keep summary row
        dt = np.median(np.diff(t_hours))
        summary_row = {"mouse": mouse_name}
        for ph in range(5):
            n_points = np.sum(labels == ph)
            summary_row[f"phase{ph}_duration_h"] = n_points * dt
        all_summaries.append(summary_row)

        # plot for each mouse
        plot_phases(mouse_name, t_hours, resid, labels, cut_times)

    # Save summary table for all mice
    df_summary = pd.DataFrame(all_summaries)
    summary_path = root / "torpor_phase_durations_all_mice.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"\nSaved phase duration summary to {summary_path.name}")


if __name__ == "__main__":
    main()
