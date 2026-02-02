import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import openpyxl

# ==========================================
# 1. 实验元数据配置
# ==========================================
DEFAULT_ROOT = Path(r"C:\Users\mayue\Desktop\Wavelet\Wavelet")
DEFAULT_OUT = Path(r"C:\Users\mayue\Desktop\Wavelet\out_v11")
FASTING_START = pd.Timestamp("2023-09-05 12:00:00") # N1-16 断食开始时间

# 训练集锚点日期 (用于让 SVM 学习休眠的特征形态)
TORPOR_ANCHOR_DAYS = {
    "22Oct2023.xlsx": ["F4", "F5", "F6"],
    "28Oct2023.xlsx": ["F3"],
    "18Nov2023.xlsx": ["F1", "F4", "F5"]
}

# ==========================================
# 2. 生理逻辑与统计标注
# ==========================================

def get_robust_baseline(series: pd.Series, q: float = 0.85) -> float:
    """计算饱腹状态基准体温，过滤环境噪声"""
    valid_s = series[(series > 33.0) & (series < 42.0)].dropna()
    if valid_s.empty: return 37.0
    return float(valid_s[valid_s >= valid_s.quantile(q)].mean())

def auto_label_by_zscore(df, z_threshold=2.0):
    """
    批判性逻辑：不使用硬阈值，使用个体统计学离群值 (Z-score) 标注。
    Z=2.0 意味着体温下降超过了该鼠日均波动的 2 倍标准差。
    """
    df['label'] = -1
    # 4Nov 作为确定的负样本 (Fed)
    df.loc[df['source_file'].str.contains('4Nov', case=False), 'label'] = 0
    
    for filename, mice_list in TORPOR_ANCHOR_DAYS.items():
        for mid in mice_list:
            mask = (df['source_file'] == filename) & (df['mouse_id'] == mid)
            if not mask.any(): continue
            m_data = df.loc[mask, 'Tb']
            # 标注 2倍标准差 以外的低体温点为休眠
            df.loc[mask & (df['Tb'] < (m_data.mean() - z_threshold * m_data.std())), 'label'] = 1
    return df

def preprocess_dataset(df: pd.DataFrame, skip_min: int = 60) -> pd.DataFrame:
    """数据预处理：切除冷启动，过滤异常年份和非生物学低温"""
    if df.empty: return pd.DataFrame()
    
    for col in ['Tb', 'HR', 'Act']:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 拦截 2038 年等时间戳噪声
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df[(df['datetime'].dt.year > 2020) & (df['datetime'].dt.year < 2030)].copy()

    processed = []
    for (mid, src), sub in df.groupby(['mouse_id', 'source_file']):
        sub = sub.sort_values('datetime').reset_index(drop=True)
        # 切除传感器戴上初期的 60 分钟爬坡期
        if len(sub) > skip_min: sub = sub.iloc[skip_min:].copy()
        # 生理拦截：低于 28.0C 视为传感器脱落，不计入统计
        sub = sub[(sub['Tb'] > 28.0) & (sub['Tb'] < 42.0)].copy()
        
        if sub.empty: continue

        # 特征提取
        sub['Tb_slope'] = sub['Tb'].diff().rolling(window=20, min_periods=1).mean().fillna(0)
        base_t = get_robust_baseline(sub['Tb'])
        sub['Tb_diff'] = sub['Tb'] - base_t
        
        if 'HR' in sub.columns and sub['HR'].notna().any():
            hr_base = sub[sub['Tb'] > (base_t - 1.2)]['HR'].median()
            sub['HR_ratio'] = sub['HR'] / (hr_base if (not pd.isna(hr_base) and hr_base > 0) else 600)
        else:
            sub['HR_ratio'] = 1.0 # N16/Synthetic 组的中性填充
            
        hours = sub['datetime'].dt.hour + sub['datetime'].dt.minute/60.0
        sub['hour_sin'] = np.sin(2 * np.pi * (hours - 12)/24.0)
        sub['hour_cos'] = np.cos(2 * np.pi * (hours - 12)/24.0)
        processed.append(sub)
    
    return pd.concat(processed) if processed else pd.DataFrame()

# ==========================================
# 3. 数据加载引擎
# ==========================================

def load_synthetic_v11(folder_path: Path):
    """修正版：直接使用 Sheet 名称作为 ID，避免单元格读取失败"""
    data = []
    for f in folder_path.glob("*.xls*"):
        xls = pd.ExcelFile(f)
        for sh_name in xls.sheet_names:
            if "summary" in sh_name.lower(): continue
            # 暴力 ID 提取：优先用 Sheet 名，这在 Synthetic 中通常就是 ID
            rfid = sh_name.strip()
            df_full = pd.read_excel(f, sheet_name=sh_name, header=None)
            h_idx = next((i for i, row in df_full.head(100).iterrows() 
                         if "Date" in str(row.values) and "Temperature" in str(row.values)), None)
            if h_idx is None: continue
            df = pd.read_excel(f, sheet_name=sh_name, header=h_idx)
            df.columns = [str(c).strip() for c in df.columns]
            t_col = [c for c in df.columns if "Temperature" in c][0]
            df['datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str), errors='coerce')
            sub = pd.DataFrame({'datetime': df['datetime'], 'Tb': pd.to_numeric(df[t_col], errors='coerce'), 
                                'mouse_id': rfid, 'source_file': f.name}).dropna(subset=['Tb', 'datetime'])
            data.append(sub)
    return pd.concat(data) if data else pd.DataFrame()

def load_natural_hr(folder_path: Path):
    data = []
    for f in folder_path.glob("*.xlsx"):
        df_raw = pd.read_excel(f, nrows=10, header=None)
        h_idx = next((i for i, row in df_raw.iterrows() if "Time Stamp" in str(row.values)), 0)
        df = pd.read_excel(f, header=h_idx)
        df.columns = [str(c).replace("\n", " ").strip() for c in df.columns]
        time_col = [c for c in df.columns if "Time" in c][0]
        mice = set(re.findall(r'([FM]\d+)', " ".join(df.columns)))
        for p in mice:
            m_cols = [c for c in df.columns if c.startswith(p)]
            sub = df[[time_col] + m_cols].copy()
            sub.columns = ['datetime'] + [c.split('.')[-1].split('_')[-1] for c in m_cols]
            sub = sub.rename(columns={'Temperature': 'Tb', 'Temp': 'Tb', 'Heart Rate': 'HR'}).assign(mouse_id=p, source_file=f.name)
            data.append(sub)
    return pd.concat(data)

# ==========================================
# 4. 可视化函数 (N16 专用)
# ==========================================

def plot_mouse_results(df, mouse_id, out_dir):
    sub = df[df['mouse_id'] == mouse_id].sort_values('datetime')
    if sub.empty: return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    
    # Tb & States
    ax1.plot(sub['datetime'], sub['Tb'], color='black', lw=1, label='Core Temp')
    ax1.fill_between(sub['datetime'], 28, 40, where=(sub['state']=='Deep'), color='red', alpha=0.3, label='Deep Torpor')
    ax1.fill_between(sub['datetime'], 28, 40, where=(sub['state']=='Gray'), color='orange', alpha=0.2, label='Gray Zone')
    ax1.axvline(FASTING_START, color='blue', linestyle='--', label='Fasting Start (05/09 12:00)')
    ax1.set_ylabel('Tb (°C)')
    ax1.set_title(f'Torpor Discovery Map: {mouse_id}')
    ax1.legend(loc='upper right')

    # Probabilities
    ax2.plot(sub['datetime'], sub['torpor_prob'], color='purple', lw=1, label='SVM Probability')
    ax2.axhline(0.7, color='red', linestyle=':', alpha=0.5)
    ax2.axhline(0.3, color='orange', linestyle=':', alpha=0.5)
    ax2.set_ylabel('Torpor Probability')
    ax2.set_xlabel('Time')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(out_dir / f"N16_{mouse_id}_Map.png", dpi=150)
    plt.close()

# ==========================================
# 5. 主程序运行
# ==========================================

def main():
    DEFAULT_OUT.mkdir(exist_ok=True)
    plots_dir = DEFAULT_OUT / "plots"
    plots_dir.mkdir(exist_ok=True)
    feats = ['Tb_diff', 'Tb_slope', 'HR_ratio', 'hour_sin', 'hour_cos']

    # --- 训练阶段 ---
    print("\n--- Phase 1: Training SVM (Z=2.0) ---")
    hr_raw = load_natural_hr(DEFAULT_ROOT / "Natural torpor HR, Core Temp, Activity")
    df_hr = preprocess_dataset(hr_raw)
    df_hr = auto_label_by_zscore(df_hr, z_threshold=2.0)
    
    train_set = df_hr[df_hr['label'] != -1].dropna(subset=feats)
    model = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel='rbf', C=1.5, class_weight='balanced', probability=True))])
    model.fit(train_set[feats], train_set['label'])
    print(f"  [OK] Model Trained. Training points: {len(train_set)}")

    # --- N16 探测与绘图 ---
    print("\n--- Phase 2: N16 Discovery ---")
    n16_folder = DEFAULT_ROOT / "Natural torpor N1-16"
    n16_raw = pd.concat([pd.read_excel(f).assign(mouse_id=f.stem, source_file=f.name) for f in n16_folder.glob("*.xlsx")])
    n16_raw = n16_raw.rename(columns={'Time': 'datetime', 'Temperature': 'Tb'})
    
    n16_proc = preprocess_dataset(n16_raw)
    n16_proc['torpor_prob'] = model.predict_proba(n16_proc[feats].fillna(0))[:, 1]
    n16_proc['state'] = np.where(n16_proc['torpor_prob'] > 0.7, 'Deep', np.where(n16_proc['torpor_prob'] > 0.3, 'Gray', 'Normal'))
    
    n16_proc.to_csv(DEFAULT_OUT / "discovery_N16_v11.csv", index=False)
    
    print("Generating Plots...")
    for mid in n16_proc['mouse_id'].unique():
        plot_mouse_results(n16_proc, mid, plots_dir)
    
    print(f"\nCompleted. Data and Plots are in {DEFAULT_OUT}")

if __name__ == "__main__":
    main()