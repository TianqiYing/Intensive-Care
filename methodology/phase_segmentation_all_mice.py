from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture


DATE_FILES = {
    "2023-10-22": "22Oct2023.xlsx",
    "2023-10-28": "28Oct2023.xlsx",
    "2023-11-04": "4Nov2023.xlsx",
    "2023-11-18": "18Nov2023.xlsx",
}


def load_day_long(path: Path, date_str: str) -> pd.DataFrame:
    # Header row may start at row 3 (index 2) as in previous scripts
    df = pd.read_excel(path, sheet_name=0, header=2, engine="openpyxl")

    records = []
    for i in range(1, 7):
        t_col = f"F{i}gpa.Temperature"
        hr_col = f"F{i}gpa.Heart Rate"
        act_col = f"F{i}gpa.Activity"
        if t_col not in df.columns:
            continue

        sub = df[["Time Stamp", t_col, hr_col, act_col]].copy()
        sub.columns = ["Time", "Temperature", "HeartRate", "Activity"]

        sub["Time"] = pd.to_datetime(sub["Time"])
        sub["Temperature"] = pd.to_numeric(sub["Temperature"], errors="coerce")
        sub["HeartRate"] = pd.to_numeric(sub["HeartRate"], errors="coerce")
        sub["Activity"] = pd.to_numeric(sub["Activity"], errors="coerce")
        sub = sub.dropna(subset=["Temperature", "HeartRate", "Activity"])

        sub["mouse"] = f"F{i}"
        sub["date_str"] = date_str
        records.append(sub)

    out = pd.concat(records, ignore_index=True)
    return out


def load_all_days(root: Path) -> pd.DataFrame:
    dfs = []
    for date_str, fname in DATE_FILES.items():
        df_day = load_day_long(root / fname, date_str)
        dfs.append(df_day)
    df_all = pd.concat(dfs, ignore_index=True)
    return df_all


def build_point_features_raw(df_all: pd.DataFrame) -> pd.DataFrame:
    feat_list = []

    for (mouse, date_str), sub in df_all.groupby(["mouse", "date_str"]):
        sub = sub.sort_values("Time").reset_index(drop=True)

        # basic numeric signals
        temp = pd.to_numeric(sub["Temperature"], errors="coerce").values.astype(float)
        hr   = pd.to_numeric(sub["HeartRate"],   errors="coerce").values.astype(float)
        act  = pd.to_numeric(sub["Activity"],    errors="coerce").values.astype(float)

        # time in seconds
        t_sec = sub["Time"].values.astype("datetime64[ns]").astype("int64") / 1e9

        n = len(sub)

        dT_dt = np.full(n, np.nan, dtype=float)
        if n > 1:
            dt_sec   = t_sec[1:] - t_sec[:-1]
            dt_hours = dt_sec / 3600.0
            for i in range(1, n):
                if dt_hours[i - 1] > 0:
                    dT_dt[i] = (temp[i] - temp[i - 1]) / dt_hours[i - 1]
                # else keep NaN

        # use local "day start" (00:00 of that date) as phase reference
        day_start = pd.to_datetime(sub["Time"].iloc[0]).normalize()
        t_hours = (sub["Time"] - day_start).dt.total_seconds() / 3600.0  # 0..24+

        period = 24.0
        phi = 2.0 * np.pi * t_hours / period

        cos_t1 = np.cos(phi)
        sin_t1 = np.sin(phi)
        cos_t2 = np.cos(2.0 * phi)
        sin_t2 = np.sin(2.0 * phi)

        feats = pd.DataFrame(
            {
                "Time":      sub["Time"],
                "temp_raw":  temp,
                "hr_raw":    hr,
                "act_raw":   act,
                "logAct_raw": np.log1p(act),
                "dT_dt":     dT_dt,
                # Fourier time features
                "t_hours":   t_hours,
                "cos_t1":    cos_t1,
                "sin_t1":    sin_t1,
                "cos_t2":    cos_t2,
                "sin_t2":    sin_t2,
                # identifiers
                "mouse":     mouse,
                "date_str":  date_str,
            }
        )

        feat_list.append(feats)

    feats_all = pd.concat(feat_list, ignore_index=True)
    return feats_all


def gmm_cluster_points(
    feats: pd.DataFrame,
    n_components_min: int = 4,
    n_components_max: int = 6,
):
    feature_cols = [
        "temp_raw",
        "hr_raw",
        "logAct_raw",
        "dT_dt",
        "cos_t1",
        "sin_t1",
        "cos_t2",
        "sin_t2",
    ]

    feats_clean = feats.copy()

    # ensure all feature columns are numeric
    for col in feature_cols:
        feats_clean[col] = pd.to_numeric(feats_clean[col], errors="coerce")

    # keep only rows where all features are finite
    mask_finite = np.isfinite(feats_clean[feature_cols]).all(axis=1)
    feats_clean = feats_clean[mask_finite].dropna(subset=feature_cols)

    print("Number of points used for GMM:", len(feats_clean))

    X = feats_clean[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_gmm = None
    best_k = None
    best_bic = np.inf

    for k in range(n_components_min, n_components_max + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=0,
        )
        gmm.fit(X_scaled)
        bic = gmm.bic(X_scaled)
        print(f"GMM K={k}, BIC={bic:.2f}")
        if bic < best_bic:
            best_bic = bic
            best_k = k
            best_gmm = gmm

    print(f"\nSelected K={best_k} with lowest BIC={best_bic:.2f}")

    labels = best_gmm.predict(X_scaled)
    feats_clean["cluster"] = labels

    # merge cluster labels back to the full feature table
    feats_out = feats.merge(
        feats_clean[["mouse", "date_str", "Time", "cluster"]],
        on=["mouse", "date_str", "Time"],
        how="left",
    )

    # quick numeric summary to understand what each cluster looks like
    summary = []
    for c in range(best_k):
        sub = feats_out[feats_out["cluster"] == c]
        if len(sub) == 0:
            continue
        summary.append(
            dict(
                cluster=c,
                n=len(sub),
                mean_T=sub["temp_raw"].mean(),
                mean_HR=sub["hr_raw"].mean(),
                mean_Act=sub["act_raw"].mean(),
                mean_dT=sub["dT_dt"].mean(),
            )
        )

    summary_df = pd.DataFrame(summary).sort_values("mean_T")
    print("\nGMM cluster summary (sorted by mean_T):")
    print(summary_df)

    return feats_out, summary_df, scaler, best_gmm, best_k


def attach_cluster_to_raw(df_all: pd.DataFrame, feats_points: pd.DataFrame) -> pd.DataFrame:
    df = df_all.copy()
    df = df.merge(
        feats_points[["mouse", "date_str", "Time", "cluster"]],
        on=["mouse", "date_str", "Time"],
        how="left",
    )
    return df


def plot_mouse_day_clusters(df_clustered: pd.DataFrame, mouse: str, date_str: str):
    df = df_clustered[
        (df_clustered["mouse"] == mouse) & (df_clustered["date_str"] == date_str)
    ].copy()
    if df.empty:
        print(f"No data for {mouse} on {date_str}")
        return

    df = df.sort_values("Time")
    t_hours = (df["Time"] - df["Time"].iloc[0]).dt.total_seconds().values / 3600.0

    plt.figure(figsize=(10, 3))
    cmap = plt.get_cmap("tab10")
    clusters = sorted(df["cluster"].dropna().unique())

    for c in clusters:
        mask = df["cluster"] == c
        plt.scatter(
            t_hours[mask],
            df["Temperature"][mask],
            s=8,
            color=cmap(int(c)),
            label=f"cluster {int(c)}",
            alpha=0.8,
        )

    plt.xlabel("Hours since day start")
    plt.ylabel("Temperature (°C)")
    plt.title(f"{mouse} on {date_str} (GMM point clusters)")
    plt.legend(fontsize=8, ncol=3)
    plt.tight_layout()
    plt.show()


def main():
    root = Path(__file__).resolve().parent

    # 1) load raw tables from four days
    df_all = load_all_days(root)
    print("All raw data shape:", df_all.shape)

    # 2) build point-level features
    feats_points = build_point_features_raw(df_all)
    print("Point feature shape:", feats_points.shape)

    # 3) GMM clustering on feature space
    feats_clustered, summary_df, scaler, gmm, K = gmm_cluster_points(
        feats_points, n_components_min=4, n_components_max=6
    )

    print("\nGMM cluster summary (sorted by mean_T):")
    print(summary_df)

    # 4) map cluster back to original time series
    df_clustered = attach_cluster_to_raw(df_all, feats_clustered)

    # 5) plot some key examples
    examples = [
        ("F4", "2023-10-22"),  # CR day with torpor
        ("F3", "2023-10-28"),  # CR day with torpor
        ("F1", "2023-11-18"),  # CR day with torpor
        ("F2", "2023-11-04"),  # full-fed baseline
    ]
    for mouse, date_str in examples:
        plot_mouse_day_clusters(df_clustered, mouse, date_str)

    # 6) save clustered raw data
    out_path = root / f"F1F6_GMM_point_clusters_K{K}.csv"
    df_clustered.to_csv(out_path, index=False)
    print(f"\nClustered raw data saved to: {out_path}")


if __name__ == "__main__":
    main()
