#!/usr/bin/env python3
"""
Generate a CSV of GEMM dimensions for a GPT-3 transformer block
under sweeps of batch size (B) and sequence length (L), then
plot how K and N change.

No external dependencies beyond: pandas, numpy, matplotlib.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------------------------------------------
# 1.  Hyper-parameters (GPT-3 block, fixed)
# ------------------------------------------------------------------
d_model = 12_288
H       = 96
d_h     = d_model // H           # 128
d_ff    = 49_152

# ------------------------------------------------------------------
# 2.  Sweep ranges (edit as needed)
# ------------------------------------------------------------------
B_list = [1, 2, 4, 8, 16]                       # batch sizes
L_list = [128, 256, 512, 1024, 2048, 4096]      # sequence lengths

# ------------------------------------------------------------------
# 3.  Generate rows
# ------------------------------------------------------------------
rows = []
for B in B_list:
    for L in L_list:
        # projections (Q, K, V)
        for name in ["Q_proj", "K_proj", "V_proj"]:
            rows.append(dict(kernel=name, B=B, L=L,
                             M=B*L, K=d_model, N=d_model))

        # attention score GEMM: Q · Kᵀ
        rows.append(dict(kernel="Q_K^T", B=B, L=L,
                         M=B*H*L, K=d_h, N=L))

        # attention context GEMM: (QKᵀ) · V
        rows.append(dict(kernel="QK_V", B=B, L=L,
                         M=B*H*L, K=L, N=d_h))

        # output projection
        rows.append(dict(kernel="Out_proj", B=B, L=L,
                         M=B*L, K=d_model, N=d_model))

        # feed-forward
        rows.append(dict(kernel="FFN_1", B=B, L=L,
                         M=B*L, K=d_model, N=d_ff))
        rows.append(dict(kernel="FFN_2", B=B, L=L,
                         M=B*L, K=d_ff,    N=d_model))

df = pd.DataFrame(rows)

# ------------------------------------------------------------------
# 4.  Save CSV
# ------------------------------------------------------------------
csv_path = Path("gemm_sweep.csv")
df.to_csv(csv_path, index=False)
print(f"[✓] CSV written to: {csv_path.resolve()}")
print(df.head())          # quick sanity-check

# ------------------------------------------------------------------
# 5.  Plot helpers
# ------------------------------------------------------------------
def plot_metric_vs_param(df_subset, param_col, value_col,
                         title, xlabel, ylabel):
    plt.figure()
    param_vals = sorted(df_subset[param_col].unique())
    for kname in df_subset['kernel'].unique():
        yvals = [df_subset[(df_subset['kernel'] == kname) &
                           (df_subset[param_col] == v)][value_col].iloc[0]
                 for v in param_vals]
        plt.plot(param_vals, yvals, label=kname)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

# ------------------------------------------------------------------
# 6.  Generate four plots
# ------------------------------------------------------------------
# (a) K vs L  (fix B = 1)
plot_metric_vs_param(df[df['B'] == 1], 'L', 'K',
                     "K dimension vs Sequence Length (B=1)",
                     "Sequence length L", "K dimension")

# (b) N vs L  (fix B = 1)
plot_metric_vs_param(df[df['B'] == 1], 'L', 'N',
                     "N dimension vs Sequence Length (B=1)",
                     "N dimension", "Sequence length L")

# (c) K vs B  (fix L = 1024)
plot_metric_vs_param(df[df['L'] == 1024], 'B', 'K',
                     "K dimension vs Batch Size (L=1024)",
                     "Batch size B", "K dimension")

# (d) N vs B  (fix L = 1024)
plot_metric_vs_param(df[df['L'] == 1024], 'B', 'N',
                     "N dimension vs Batch Size (L=1024)",
                     "Batch size B", "N dimension")

plt.show()
