# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 12:47:56 2025

@author: harry
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load cleaned data from earlier steps
file_path = '4Nov2023.xlsx'

# Read properly: header row is row 1, actual headers in row 2
df0 = pd.read_excel(file_path, header=1)
new_cols = df0.iloc[0]
df = df0[1:].copy()
df.columns = new_cols

# Convert time
df['Time Stamp'] = pd.to_datetime(df['Time Stamp'])

# Convert numeric
for col in df.columns:
    if col != 'Time Stamp':
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Standardize each parameter so shapes can be compared
scaler = StandardScaler()

mice = ['F1gpa','F2gpa','F3gpa','F4gpa','F5gpa','F6gpa']

for mouse in mice:
    cols = [f"{mouse}.Temperature", f"{mouse}.Heart Rate", f"{mouse}.Activity"]
    sub_df = df[cols]

    # Scale (ignore NaNs)
    scaled = pd.DataFrame(scaler.fit_transform(sub_df.fillna(method='ffill').fillna(method='bfill')),
                          columns=cols)

    plt.figure(figsize=(12,5))
    for col in cols:
        plt.plot(df['Time Stamp'], scaled[col], label=col)

    plt.title(f"{mouse} – Scaled to fit")
    plt.xlabel("Time")
    plt.ylabel("Scaled Value (mean=0, std=1)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{mouse}_normalized_plot.png", dpi=300, bbox_inches="tight")
    plt.show()
