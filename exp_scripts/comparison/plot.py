import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
parser = argparse.ArgumentParser(description="Plot speedup bar chart from latency CSV")
parser.add_argument("--input", type=str, default="latencies_10-3.csv", help="Path to the input CSV file")
args = parser.parse_args()
# Load your CSV
df = pd.read_csv(args.input)

# Row 0 = baseline latencies, Row 1 = new latencies
baseline = df.iloc[0].values
new = df.iloc[1].values
workloads = df.columns.tolist()

# Compute speedups (baseline / new)
speedups = baseline / new

# Plot
plt.figure(figsize=(5, 4))
x = np.arange(len(workloads))
bars = plt.bar(x, speedups)

# Labels and styling for academic paper
plt.xticks(x, workloads, rotation=45, ha="right", fontsize=9)
plt.yscale("log")
plt.ylabel("Speedup (×)", fontsize=9)
plt.title("Workload Speedup vs. H100 GPU", fontsize=9)

# Add values above bars
for bar, val in zip(bars, speedups):
    plt.text(
        bar.get_x() + bar.get_width() / 2, 
        bar.get_height(), 
        f"{val:.2f}", 
        ha="center", va="bottom", fontsize=8
    )

plt.tight_layout()
plt.savefig("speedup_bar_chart.pdf")  # for paper use
plt.show()