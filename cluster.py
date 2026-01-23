
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import umap
import matplotlib.pyplot as plt

# 1. Load data
path = r"C:\Users\mayue\Desktop\Wavelet\demo.xlsx"  
df = pd.read_excel(path)

# 2. 定义特征：
#   - direct KMeans: 温度 + activity
#   - UMAP: 温度 + 心率 + activity
direct_cols = ["F6gpa.Temperature", "F6gpa.Activity"]
umap_cols   = ["F6gpa.Temperature", "F6gpa.Heart Rate", "F6gpa.Activity"]

# 转成数值类型
for col in set(direct_cols + umap_cols):
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 公共 mask：三列都不是 NaN 的行
mask = df[umap_cols].notna().all(axis=1)

# 提取 KMeans 用的特征：Temp + Activity（2 维）
X_direct = df.loc[mask, direct_cols].values

# 提取 UMAP 用的特征：Temp + HR + Activity（3 维）
X_umap_in = df.loc[mask, umap_cols].values

print("X_direct shape (Temp+Activity):", X_direct.shape)   # (N, 2)
print("X_umap_in shape (T+HR+Act):   ", X_umap_in.shape)   # (N, 3)

time = df.loc[mask, "Time Stamp"].values if "Time Stamp" in df.columns else None

# 3. 标准化
scaler_direct = StandardScaler()
X_direct_scaled = scaler_direct.fit_transform(X_direct)

scaler_umap = StandardScaler()
X_umap_scaled = scaler_umap.fit_transform(X_umap_in)

# 4. 直接 KMeans：只看 Temperature + Activity
k = 3  # 想要几类就改这里
kmeans_direct = KMeans(n_clusters=k, random_state=0, n_init=10)
labels_direct = kmeans_direct.fit_predict(X_direct_scaled)

df.loc[mask, "cluster_direct_TA"] = labels_direct

print("Direct KMeans (Temp+Activity) cluster sizes:", np.bincount(labels_direct))
print("Direct centers (z-score, [Temp, Activity]):")
print(kmeans_direct.cluster_centers_)

# 5. UMAP：用 3 个特征先降维，再单独 KMeans（标签仍然算出来）
reducer = umap.UMAP(
    n_neighbors=30,
    min_dist=0.1,
    metric="euclidean",
    n_components=2,
    random_state=0,
)
X_emb = reducer.fit_transform(X_umap_scaled)
print("UMAP embedding shape:", X_emb.shape)  # (N, 2)

kmeans_umap = KMeans(n_clusters=k, random_state=0, n_init=10)
labels_umap = kmeans_umap.fit_predict(X_emb)

df.loc[mask, "cluster_umap_3feat"] = labels_umap
print("UMAP+KMeans cluster sizes:", np.bincount(labels_umap))

# 6. 画图：颜色统一用 labels_direct
cmap = plt.get_cmap("tab10")

# (1) 直接 KMeans：特征空间（Temp, Activity）
plt.figure()
plt.scatter(
    X_direct_scaled[:, 0],
    X_direct_scaled[:, 1],
    c=labels_direct,
    s=8,
    cmap=cmap,
    vmin=0,
    vmax=k-1,
)
plt.xlabel("Temperature (z)")
plt.ylabel("Activity (z)")
plt.title("Direct KMeans on Temperature + Activity")
plt.tight_layout()

# (2) UMAP 嵌入空间，用同一套 labels_direct 上色
plt.figure()
plt.scatter(
    X_emb[:, 0],
    X_emb[:, 1],
    c=labels_direct,   # 关键：颜色按 direct KMeans 的标签
    s=8,
    cmap=cmap,
    vmin=0,
    vmax=k-1,
)
plt.xlabel("UMAP-1 (T+HR+Act)")
plt.ylabel("UMAP-2")
plt.title("UMAP (T+HR+Act), colored by direct KMeans clusters")
plt.tight_layout()

plt.show()

