import os
import re
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# ==========================================
# 1. SCIENTIFIC ANCHORS (From Key.docx)
# ==========================================
TORPOR_ANCHORS = [
    {"file": "22Oct2023", "mice": ["F1", "F3", "F4", "F6"]},
    {"file": "28Oct2023", "mice": ["F3", "F4"]},
    {"file": "18Nov2023", "mice": ["F3", "F4", "F5", "F6"]}
]

def smart_load_excel(file_path):
    """Parses Excel with complex headers and extracts long-format data."""
    print(f"  Parsing: {file_path.name}")
    preview = pd.read_excel(file_path, nrows=10, header=None)
    header_idx = 0
    for i, row in preview.iterrows():
        if "Time Stamp" in str(row.values):
            header_idx = i
            break
            
    df = pd.read_excel(file_path, header=header_idx)
    df.columns = [str(c).replace("\n", " ").strip() for c in df.columns]
    df = df[df.iloc[:, 0].astype(str).str.contains(r'\d', na=False)]
    
    time_col = [c for c in df.columns if "Time" in c][0]
    df['datetime'] = pd.to_datetime(df[time_col])
    
    long_list = []
    prefixes = set(re.findall(r'([FM]\d+)', " ".join(df.columns)))
    
    for p in prefixes:
        m_cols = [c for c in df.columns if c.startswith(p)]
        if not m_cols: continue
        sub = df[[time_col] + m_cols].copy()
        mapping = {time_col: 'datetime'}
        for c in m_cols:
            if 'Temperature' in c: mapping[c] = 'Tb'
            elif 'Heart Rate' in c: mapping[c] = 'HR'
            elif 'Activity' in c: mapping[c] = 'Act'
        sub = sub.rename(columns=mapping).assign(mouse_id=p, source_file=file_path.name)
        for col in ['Tb', 'HR', 'Act']:
            if col in sub.columns: sub[col] = pd.to_numeric(sub[col], errors='coerce')
        long_list.append(sub)
    return pd.concat(long_list)

def process_features(df):
    """Extracts data-driven features for SVM boundary discovery."""
    df = df.sort_values(['mouse_id', 'datetime'])
    base_data = df[df['source_file'].str.contains('4Nov', case=False) & (df['Tb'] > 33)]
    tb_baselines = base_data.groupby('mouse_id')['Tb'].median().to_dict()
    hr_baselines = base_data.groupby('mouse_id')['HR'].median().to_dict()
    
    results = []
    for mid, group in df.groupby('mouse_id'):
        group = group.copy()
        b_t = tb_baselines.get(mid, group[group['Tb'] > 33]['Tb'].head(60).median())
        b_h = hr_baselines.get(mid, group[group['HR'] > 100]['HR'].head(60).median())
        
        group['Tb_diff'] = group['Tb'] - b_t
        group['Tb_rate'] = group['Tb'].diff().fillna(0) # Slope: critical for sleep vs torpor
        group['HR_ratio'] = group['HR'] / b_h
        
        hours = group['datetime'].dt.hour + group['datetime'].dt.minute/60.0
        group['hour_sin'] = np.sin(2 * np.pi * (hours - 12)/24)
        group['hour_cos'] = np.cos(2 * np.pi * (hours - 12)/24)
        results.append(group)
    return pd.concat(results)

def run_svm_discovery(root_path):
    root = Path(root_path) / "Natural torpor HR, Core Temp, Activity"
    out_dir = Path("output_svm_discovery")
    out_dir.mkdir(exist_ok=True)

    # 1. Prepare Data
    raw_data = pd.concat([smart_load_excel(f) for f in root.glob("*.xlsx")])
    df = process_features(raw_data)

    # 2. Train-Inference Labels (0: Baseline, 1: Anchors, -1: Undefined)
    df['training_label'] = -1
    df.loc[df['source_file'].str.contains('4Nov'), 'training_label'] = 0
    for event in TORPOR_ANCHORS:
        mask = (df['source_file'].str.contains(event['file'])) & \
               (df['mouse_id'].isin(event['mice'])) & (df['Tb_diff'] < -3.5)
        df.loc[mask, 'training_label'] = 1

    # 3. SVM Training
    feat_cols = ['Tb_diff', 'Tb_rate', 'HR_ratio', 'hour_sin', 'hour_cos']
    train_set = df[df['training_label'] != -1].dropna(subset=feat_cols)
    X_train, y_train = train_set[feat_cols], train_set['training_label']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    svm = SVC(kernel='rbf', C=1.5, class_weight='balanced', probability=True)
    svm.fit(X_scaled, y_train)

    # 4. Inference & Gray Zone Discovery
    X_all_scaled = scaler.transform(df[feat_cols].fillna(0))
    df['torpor_prob'] = svm.predict_proba(X_all_scaled)[:, 1]
    
    # Categorize states based on probability (Discovery logic)
    df['discovered_state'] = 'Normal'
    df.loc[df['torpor_prob'] > 0.7, 'discovered_state'] = 'Deep Torpor'
    df.loc[(df['torpor_prob'] <= 0.7) & (df['torpor_prob'] >= 0.3), 'discovered_state'] = 'Gray Zone'
    
    # 5. Analysis of Gray Zones (Transition states)
    gray_zones = df[df['discovered_state'] == 'Gray Zone']
    if not gray_zones.empty:
        summary = gray_zones.groupby('source_file').agg({
            'Tb_diff': 'mean', 'HR_ratio': 'mean', 'Tb_rate': 'mean'
        })
        summary.to_csv(out_dir / "gray_zone_analysis.csv")
        print("\n--- Gray Zone Analysis (Undefined States) ---")
        print(summary)

    df.to_csv(out_dir / "full_discovery_results.csv", index=False)
    
    # Visualization: Example of Gray Zone Discovery
    mid = df['mouse_id'].unique()[0]
    sample = df[df['mouse_id'] == mid].sort_values('datetime')
    plt.figure(figsize=(12, 6))
    plt.plot(sample['datetime'], sample['Tb'], color='black', alpha=0.3, label='Tb Curve')
    plt.scatter(sample['datetime'], sample['Tb'], c=sample['torpor_prob'], cmap='coolwarm', s=10)
    plt.colorbar(label='SVM Torpor Probability')
    plt.title(f"Discovery Map: Mouse {mid} (Red=Deep, White=Gray Zone, Blue=Normal)")
    plt.savefig(out_dir / f"probability_map_{mid}.png")
    
    print(f"\nPipeline complete. Files saved in '{out_dir}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=r"C:\Users\mayue\Desktop\Wavelet\Wavelet")
    args = parser.parse_args()
    run_svm_discovery(args.root)