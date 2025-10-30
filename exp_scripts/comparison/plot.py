import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Plot grouped speedup bar chart from latency CSV")
parser.add_argument("--input", type=str, default="latencies_10-3.csv", help="Path to the input CSV file")
args = parser.parse_args()

# Load CSV
df = pd.read_csv(args.input)

# Row 0 = baseline latencies, Row 1 = Proteus latencies, Row 2 = new latencies
baseline = df.iloc[0].values
proteus = df.iloc[1].values
new = df.iloc[2].values
workloads = df.columns.tolist()

# Compute speedups (baseline / latency)
speedups_proteus = baseline / proteus
speedups_new = baseline / new

# Plot
plt.figure(figsize=(6, 5))

x = np.arange(len(workloads))
width = 0.25  # width of each bar

bars1 = plt.bar(x - width, np.ones_like(baseline), width, label="H100", color="lightgray", edgecolor="black")
bars2 = plt.bar(x, speedups_proteus, width, label="Proteus", color="#7fc97f", edgecolor="black")
bars3 = plt.bar(x + width, speedups_new, width, label="ReAP(Ours)", color="#beaed4", edgecolor="black")

# Labels and style
plt.xticks(x, workloads, rotation=45, ha="right", fontsize=9)
plt.yscale("log")
plt.ylabel("Speedups vs. H100 (×)", fontsize=9)
plt.title("Workload Speedup Comparison", fontsize=9)
plt.legend(fontsize=8, frameon=False)

# Add annotations
for bars in [bars3]:
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2, 
            height * 1.05, 
            f"{height:.2f}", 
            ha="center", va="bottom", fontsize=7
        )

plt.tight_layout()
plt.savefig("speedup_bar_chart.pdf")
plt.show()
