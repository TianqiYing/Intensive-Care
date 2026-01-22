import os
import re
import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ==========================================
# 1. FIXED LOCAL PATHS
# ==========================================
DEFAULT_ROOT = Path(r"C:\Users\mayue\Desktop\Wavelet\Wavelet")
DEFAULT_OUT = Path(r"C:\Users\mayue\Desktop\Wavelet\out_v8")

TORPOR_ANCHORS = [
    {"file": "22Oct2023", "mice": ["F1", "F3", "F4", "F6"]},
    {"file": "28Oct2023", "mice": ["F3", "F4"]},
    {"file": "18Nov2023", "mice": ["F3", "F4", "F5", "F6"]},
]

# ==========================================
# 2. PHYSIOLOGICAL LOGIC
# ==========================================

def get_robust_baseline(series: pd.Series, q: float = 0.8) -> float:
    valid_s = series[series > 33.0].dropna()
    if valid_s.empty: return 37.0
    threshold = valid_s.quantile(q)
    return float(valid_s[valid_s >= threshold].mean())

def preprocess_dataset(df: pd.DataFrame, skip_min: int = 60) -> pd.DataFrame:
    if df.empty: return pd.DataFrame()
    
    # Force numeric
    for col in ['Tb', 'HR', 'Act']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    processed = []
    # Ensure source_file exists
    if 'source_file' not in df.columns: df['source_file'] = 'unknown'
    
    for (mid, src), sub in df.groupby(['mouse_id', 'source_file']):
        sub = sub.sort_values('datetime').reset_index(drop=True)
        if len(sub) > skip_min: sub = sub.iloc[skip_min:].copy()
        if sub['Tb'].dropna().empty: continue

        sub['Tb_slope'] = sub['Tb'].diff().rolling(window=20, min_periods=1).mean().fillna(0)
        base_t = get_robust_baseline(sub['Tb'])
        sub['Tb_diff'] = sub['Tb'] - base_t
        
        if 'HR' in sub.columns and sub['HR'].notna().any():
            hr_base = sub[sub['Tb'] > (base_t - 1.5)]['HR'].median()
            sub['HR_ratio'] = sub['HR'] / (hr_base if (not pd.isna(hr_base) and hr_base > 0) else 600)
        else:
            sub['HR_ratio'] = 1.0 
            
        hours = sub['datetime'].dt.hour + sub['datetime'].dt.minute/60.0
        sub['hour_sin'] = np.sin(2 * np.pi * (hours - 12)/24.0)
        sub['hour_cos'] = np.cos(2 * np.pi * (hours - 12)/24.0)
        processed.append(sub)
    
    return pd.concat(processed) if processed else pd.DataFrame()

# ==========================================
# 3. LOADERS
# ==========================================

def load_natural_hr(root: Path):
    folder = root / "Natural torpor HR, Core Temp, Activity"
    data = []
    for f in folder.glob("*.xlsx"):
        preview = pd.read_excel(f, nrows=10, header=None)
        h_idx = 0
        for i, row in preview.iterrows():
            if "Time Stamp" in str(row.values): h_idx = i; break
        df = pd.read_excel(f, header=h_idx)
        df.columns = [str(c).replace("\n", " ").strip() for c in df.columns]
        time_col = [c for c in df.columns if "Time" in c][0]
        dt = pd.to_datetime(df[time_col], errors='coerce')
        mice = set(re.findall(r'([FM]\d+)', " ".join(df.columns)))
        for p in mice:
            m_cols = [c for c in df.columns if c.startswith(p)]
            sub = df[[time_col] + m_cols].copy()
            sub.columns = ['datetime'] + [c.split('.')[-1].split('_')[-1] for c in m_cols]
            mapping = {'Temperature': 'Tb', 'Temp': 'Tb', 'Heart Rate': 'HR', 'Activity': 'Act'}
            sub = sub.rename(columns=mapping).assign(mouse_id=p, source_file=f.name)
            sub['datetime'] = dt
            data.append(sub)
    return pd.concat(data)

def load_natural_n(root: Path):
    folder = root / "Natural torpor N1-16"
    data = []
    for f in folder.glob("*.xlsx"):
        df = pd.read_excel(f)
        df.columns = [str(c).strip() for c in df.columns]
        df = df.rename(columns={'Time': 'datetime', 'Temperature': 'Tb'})
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['mouse_id'] = f.stem
        df['source_file'] = f.name # Fix: Ensure this column is here
        data.append(df[['datetime', 'mouse_id', 'Tb', 'source_file']])
    return pd.concat(data)

def load_synthetic(root: Path):
    folder = root / "Synthetic torpor"
    data = []
    # 匹配 .xlsx 或 .xls 文件
    files = list(folder.glob("*.xls*"))
    for f in files:
        print(f"  Attempting Synthetic: {f.name}")
        xls = pd.ExcelFile(f)
        for sh in xls.sheet_names:
            if "summary" in sh.lower(): continue
            
            # 读取前 50 行来精确定位 Snapshots 数据区
            df_full = pd.read_excel(xls, sheet_name=sh, header=None)
            h_idx = None
            for i, row in df_full.iterrows():
                row_str = " ".join(row.astype(str))
                # 寻找 Synthetic 特有的标志位：Snapshots 或 Date+Time+Temperature
                if "Date" in row_str and "Time" in row_str and "Temperature" in row_str:
                    h_idx = i
                    break
            
            if h_idx is None:
                continue
                
            # 重新以正确行作为表头读取
            df = pd.read_excel(xls, sheet_name=sh, header=h_idx)
            df.columns = [str(c).strip() for c in df.columns]
            
            # 找到关键列（使用更宽泛的正则匹配）
            t_col = [c for c in df.columns if "Temperature" in c]
            a_col = [c for c in df.columns if "Activity" in c]
            d_col = [c for c in df.columns if "Date" in c]
            ti_col = [c for c in df.columns if "Time" in c]
            
            if not (t_col and d_col and ti_col):
                continue
            
            # 合并日期和时间
            df['datetime'] = pd.to_datetime(df[d_col[0]].astype(str) + ' ' + df[ti_col[0]].astype(str), errors='coerce')
            
            # 提取 RFID (通常在元数据区，或者在列里)
            # 尝试从前面的元数据行提取 RFID
            rfid = "unknown"
            for i in range(h_idx):
                row_str = " ".join(df_full.iloc[i].astype(str))
                if "RFID" in row_str:
                    rfid = row_str.split(":")[-1].strip() if ":" in row_str else row_str.split(" ")[-1].strip()
                    break

            sub = pd.DataFrame({
                'datetime': df['datetime'],
                'Tb': pd.to_numeric(df[t_col[0]], errors='coerce'),
                'Act': pd.to_numeric(df[a_col[0]], errors='coerce') if a_col else 0.0,
                'mouse_id': rfid if rfid != "unknown" else sh,
                'source_file': f.name
            })
            
            data.append(sub.dropna(subset=['datetime', 'Tb']))
            
    return pd.concat(data) if data else pd.DataFrame()

# ==========================================
# 4. MAIN
# ==========================================

def main():
    DEFAULT_OUT.mkdir(exist_ok=True)
    feats = ['Tb_diff', 'Tb_slope', 'HR_ratio', 'hour_sin', 'hour_cos']

    print("\n--- Phase 1: Training ---")
    df_hr = preprocess_dataset(load_natural_hr(DEFAULT_ROOT))
    df_hr['train_label'] = -1
    df_hr.loc[df_hr['source_file'].str.contains('4Nov'), 'train_label'] = 0
    for anchor in TORPOR_ANCHORS:
        mask = (df_hr['source_file'].str.contains(anchor['file'])) & \
               (df_hr['mouse_id'].isin(anchor['mice'])) & (df_hr['Tb_diff'] < -3.5)
        df_hr.loc[mask, 'train_label'] = 1
    
    train_set = df_hr[df_hr['train_label'] != -1].dropna(subset=feats)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_set[feats])
    model = SVC(kernel='rbf', C=1.5, class_weight='balanced', probability=True)
    model.fit(X_train, train_set['train_label'])
    
    # Discovery
    for mode, loader in [("N16", load_natural_n), ("Synthetic", load_synthetic)]:
        print(f"\n--- Phase 2: {mode} ---")
        raw = loader(DEFAULT_ROOT)
        if raw.empty: print(f"  [Skip] No data found for {mode}"); continue
        proc = preprocess_dataset(raw)
        if 'HR_ratio' not in proc.columns: proc['HR_ratio'] = 1.0
        X_test = scaler.transform(proc[feats].fillna(0))
        proc['torpor_prob'] = model.predict_proba(X_test)[:, 1]
        proc.to_csv(DEFAULT_OUT / f"discovery_{mode}.csv", index=False)
        print(f"  [OK] Saved {mode}")

if __name__ == "__main__":
    main()