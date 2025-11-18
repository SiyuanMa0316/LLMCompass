#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# === Parameters ===
CSV_FILE = "mapping_sensitivity.csv"
TRIM_PERCENT = 0.10        # keep <= 10th percentile
BINS = 80
SAVE_PNG = "latency_distribution_trim10.pdf"

# === Load & clean ===
df = pd.read_csv(CSV_FILE)
df["latency"] = pd.to_numeric(df["latency"], errors="coerce")
df = df.dropna(subset=["latency"])

# === Percentiles ===
p5 = df["latency"].quantile(0.05)
p10 = df["latency"].quantile(0.10)

# === Trim ===
trim_cut = p10
trimmed = df[df["latency"] <= trim_cut].copy()

def nearest_row_by_latency(dataframe, target):
    diffs = (dataframe["latency"] - target).abs()
    idx = diffs.sort_values(kind="mergesort").index[0]
    return dataframe.loc[idx]

best_row = trimmed.loc[trimmed["latency"].idxmin()]
p5_row   = nearest_row_by_latency(trimmed, p5)
p10_row  = nearest_row_by_latency(trimmed, p10)

print("=== Key Mappings ===")
print(f"Best: {best_row['tile_mapping']} | {best_row['arr_mapping']} | {best_row['latency']:.4g}")
print(f"5th pct: {p5_row['tile_mapping']} | {p5_row['arr_mapping']} | {p5_row['latency']:.4g}")
print(f"10th pct: {p10_row['tile_mapping']} | {p10_row['arr_mapping']} | {p10_row['latency']:.4g}")

# === Plot ===
plt.figure(figsize=(3.5, 2.3))  # fits one column (~3.5 in width)
plt.hist(trimmed["latency"], bins=BINS, edgecolor="black", alpha=0.75)
plt.xlabel("Latency", fontsize=10)
plt.ylabel("Count", fontsize=10)
plt.title("Latency Distribution (≤10th percentile)", fontsize=10)
plt.grid(True, linestyle="--", alpha=0.5)

# === Vertical markers with larger text ===
def vline(x, label, side="left"):
    plt.axvline(x, linestyle="--", linewidth=1)
    ymin, ymax = plt.ylim()
    y = ymax * 0.93
    if side == "right":
        plt.text(x, y, label, rotation=90, va="top", ha="left", fontsize=8)
    else:
        plt.text(x, y, label, rotation=90, va="top", ha="right", fontsize=8)

# move 'best' marker to right
vline(best_row["latency"],
      f"Best\n{best_row['tile_mapping']}\n{best_row['arr_mapping']}\n{best_row['latency']:.3g}",
      side="right")
vline(p5_row["latency"],
      f"5th pct\n{p5_row['tile_mapping']}\n{p5_row['arr_mapping']}\n{p5_row['latency']:.3g}")
vline(p10_row["latency"],
      f"10th pct\n{p10_row['tile_mapping']}\n{p10_row['arr_mapping']}\n{p10_row['latency']:.3g}")

plt.tight_layout(pad=0.4)

if SAVE_PNG:
    Path(SAVE_PNG).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(SAVE_PNG, dpi=300, bbox_inches="tight")

plt.show()
