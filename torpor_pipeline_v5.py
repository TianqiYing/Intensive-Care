import os
import re
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ==========================================
# 1. 核心实验配置 (基于 Key.docx)
# ==========================================
# 包含光照周期、关键实验日期以及是否具备心率/活动度数据
CONFIG = {
    "hr": {
        "subfolder": "Natural torpor HR, Core Temp, Activity",
        "baseline_file": "4Nov2023", # Fed Baseline
        "lights_on": 12,
        "features": ['Tb_diff', 'Tb_rate', 'hour_sin', 'hour_cos', 'HR_ratio', 'Act']
    },
    "n16": {
        "subfolder": "Natural torpor N1-16",
        "baseline_file": None, # 每个文件自带基准
        "lights_on": 12,
        "features": ['Tb_diff', 'Tb_rate', 'hour_sin', 'hour_cos']
    },
    "synthetic": {
        "subfolder": "Synthetic torpor",
        "baseline_file": "Z-batch", # Z-batch 包含 24h 基准
        "lights_on": 10,
        "features": ['Tb_diff', 'Tb_rate', 'hour_sin', 'hour_cos', 'Act']
    }
}

# ==========================================
# 2. 智能解析引擎
# ==========================================

def smart_load(file_path, mode):
    """解析不同格式的 Excel，处理复杂的复合表头"""
    print(f"  解析中: {file_path.name}")
    
    # 针对 hr 模式的特殊处理 (如 18Nov2023.xlsx)
    if mode == "hr":
        # 读取前几行来探测真正的表头位置
        df_preview = pd.read_excel(file_path, nrows=5, header=None)
        header_row = 0
        for i, row in df_preview.iterrows():
            row_str = " ".join(row.astype(str).values)
            if "Time Stamp" in row_str:
                header_row = i # 找到 "Time Stamp" 所在的行索引
                break
        
        # 重新读取，以找到的行作为 header
        df = pd.read_excel(file_path, header=header_row)
        # 清理列名，去除可能存在的换行符
        df.columns = [str(c).replace("\n", " ").strip() for c in df.columns]
        
        # 过滤掉第 0 位置可能残留的表头重读行（有些文件 Time Stamp 下一行可能还是文字）
        df = df[df.iloc[:, 0].astype(str).str.contains(r'\d', na=False)]
        
        time_col = [c for c in df.columns if "Time Stamp" in c or "Time" in c][0]
        df['datetime'] = pd.to_datetime(df[time_col])
        
        mice_data = []
        # 提取 F1, F2 等小鼠前缀
        prefixes = set(re.findall(r'([FM]\d+)', " ".join(df.columns)))
        for p in prefixes:
            # 找到属于该小鼠的所有列 (Temp, HR, Act)
            m_cols = [c for c in df.columns if c.startswith(p)]
            if not m_cols: continue
            
            sub = df[[time_col] + m_cols].copy()
            # 简化列名：F1gpa.Temperature -> Temperature
            new_cols = {time_col: 'datetime'}
            for c in m_cols:
                if 'Temperature' in c: new_cols[c] = 'Tb'
                elif 'Heart Rate' in c: new_cols[c] = 'HR'
                elif 'Activity' in c: new_cols[c] = 'Act'
            
            sub = sub.rename(columns=new_cols)
            sub['mouse_id'] = p
            # 强制转换为数值，非数值设为 NaN
            for col in ['Tb', 'HR', 'Act']:
                if col in sub.columns:
                    sub[col] = pd.to_numeric(sub[col], errors='coerce')
            
            mice_data.append(sub)
        return pd.concat(mice_data)

    # N16 和 Synthetic 模式保持之前 v5 的逻辑
    elif mode == "n16":
        df = pd.read_excel(file_path)
        df.columns = [str(c).strip() for c in df.columns]
        df['datetime'] = pd.to_datetime(df['Time'])
        df = df.rename(columns={"Temperature": "Tb"})
        df['mouse_id'] = file_path.stem
        df['Tb'] = pd.to_numeric(df['Tb'], errors='coerce')
        return df[['datetime', 'mouse_id', 'Tb']]
    
    # ... (Synthetic 逻辑保持不变)

# ==========================================
# 3. 鲁棒基线与高级特征 (区分睡眠与休眠)
# ==========================================

def calculate_advanced_features(df, mode, root_path):
    """
    计算稳健基线并提取 SVM 特征。
    排除传感器初始化时的低温异常值 (>33°C 视为有效)。
    """
    df = df.sort_values(['mouse_id', 'datetime']).reset_index(drop=True)
    
    # 1. 建立稳健基准线
    baselines = {}
    if mode == "hr":
        # 强制使用 4Nov2023 作为所有个体的 Fed Baseline
        base_df = df[(df['source_file'].str.contains('4Nov', case=False)) & (df['Tb'] > 33)]
        baselines = base_df.groupby('mouse_id')['Tb'].median().to_dict()
        hr_baselines = base_df.groupby('mouse_id')['HR'].median().to_dict()
    
    # 2. 应用特征
    processed_list = []
    for mid, group in df.groupby('mouse_id'):
        group = group.copy()
        # 如果没有全局基准，则取每只鼠前 6h 的有效均值
        base_t = baselines.get(mid, group[group['Tb'] > 33]['Tb'].head(360).median())
        
        group['Tb_diff'] = group['Tb'] - base_t
        group['Tb_rate'] = group['Tb'].diff().fillna(0) # 区分休眠的剧烈降温与睡眠的平缓降温
        
        if 'HR' in group.columns:
            base_hr = hr_baselines.get(mid, group['HR'].head(360).median())
            group['HR_ratio'] = group['HR'] / base_hr

        # 循环时间特征：基于开灯时间校准相位
        lights_on = CONFIG[mode]['lights_on']
        hours = group['datetime'].dt.hour + group['datetime'].dt.minute/60.0
        rel_hours = (hours - lights_on) % 24
        group['hour_sin'] = np.sin(2 * np.pi * rel_hours / 24)
        group['hour_cos'] = np.cos(2 * np.pi * rel_hours / 24)
        
        processed_list.append(group)
    
    return pd.concat(processed_list)

# ==========================================
# 4. 主流水线：训练 SVM 并预测
# ==========================================

def run_v5_pipeline(mode, root_path):
    conf = CONFIG[mode]
    data_dir = Path(root_path) / conf['subfolder']
    
    # 加载数据
    files = list(data_dir.glob("*.xlsx"))
    dfs = []
    for f in files:
        df_tmp = smart_load(f, mode)
        df_tmp['source_file'] = f.name
        dfs.append(df_tmp)
    
    full_df = pd.concat(dfs)
    
    # 特征工程
    full_df = calculate_advanced_features(full_df, mode, root_path)
    
    # 自动标注用于训练 SVM (休眠定义：温差 < -4.5 且心率大幅下降)
    if 'HR_ratio' in full_df.columns:
        full_df['label'] = ((full_df['Tb_diff'] < -4.5) & (full_df['HR_ratio'] < 0.75)).astype(int)
    else:
        full_df['label'] = (full_df['Tb_diff'] < -4.5).astype(int)

    # SVM 训练
    feat_cols = [c for c in conf['features'] if c in full_df.columns]
    X = full_df[feat_cols].fillna(0)
    y = full_df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    
    # 使用 balanced 权重应对样本不平衡（休眠时间远少于非休眠时间）
    clf = SVC(kernel='rbf', class_weight='balanced', probability=True)
    clf.fit(X_train_s, y_train)
    
    print(f"\n--- SVM 评估结果 ({mode}) ---")
    y_pred = clf.predict(scaler.transform(X_test))
    print(classification_report(y_test, y_pred))

    # 保存与绘图
    out_path = Path("output_final")
    out_path.mkdir(exist_ok=True)
    full_df.to_csv(out_path / f"results_{mode}.csv", index=False)
    
    # 绘制一只典型小鼠的检测图
    mid = full_df['mouse_id'].unique()[0]
    sample = full_df[full_df['mouse_id'] == mid].sort_values('datetime')
    plt.figure(figsize=(14, 6))
    plt.plot(sample['datetime'], sample['Tb'], label='Core Temp', color='blue')
    plt.fill_between(sample['datetime'], 20, 40, where=sample['label']==1, color='red', alpha=0.2, label='Detected Torpor')
    plt.title(f"SVM Detection: Mouse {mid} ({mode} mode)")
    plt.legend()
    plt.savefig(out_path / f"plot_{mode}_{mid}.png")
    print(f"完成！检查 output_final 文件夹。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['hr', 'n16', 'synthetic'], default='hr')
    parser.add_argument("--root", type=str, default=r"C:\Users\mayue\Desktop\Wavelet\Wavelet")
    args = parser.parse_args()
    run_v5_pipeline(args.mode, args.root)