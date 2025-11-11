import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Plot grouped speedup bar chart from latency CSV")
parser.add_argument("--input", type=str, default="latencies_run_all_output_10-9.csv", help="Path to the input CSV file")
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
plt.figure(figsize=(8, 6))

x = np.arange(len(workloads))
width = 0.25  # width of each bar

bars1 = plt.bar(x - width, np.ones_like(baseline), width, label="H100", color="#b8b0b0", edgecolor="black")
bars2 = plt.bar(x, speedups_proteus, width, label="Proteus", color="#ef7b7b", edgecolor="black")
bars3 = plt.bar(x + width, speedups_new, width, label="ReMAP(Ours)", color="#82b4e7", edgecolor="black")



# Labels and style
plt.xticks(x, workloads, rotation=30, ha="right", fontsize=12)
plt.yscale("log")
plt.ylabel("Speedup normalzied to H100", fontsize=15)
# plt.title("Workload Speedup Comparison", fontsize=15)
plt.axhline(y=1, color="black", linestyle="--", linewidth=1, alpha=0.7)
plt.legend(fontsize=10, frameon=False)

# Add annotations
for bars in [bars3]:
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2, 
            height * 1.05, 
            f"{height:.2f}", 
            ha="center", va="bottom", fontsize=12
        )

        # Add geomean bar
        geomean_baseline = np.exp(np.mean(np.log(baseline)))
        geomean_proteus = np.exp(np.mean(np.log(speedups_proteus)))
        geomean_new = np.exp(np.mean(np.log(speedups_new)))

        x_geomean = len(workloads)
        plt.bar(x_geomean - width, 1, width, color="#b8b0b0", edgecolor="black")
        plt.bar(x_geomean, geomean_proteus, width, color="#ef7b7b", edgecolor="black")
        plt.bar(x_geomean + width, geomean_new, width, color="#82b4e7", edgecolor="black")

        plt.xticks(list(x) + [x_geomean], workloads + ["geomean"], rotation=30, ha="right", fontsize=12)

        plt.text(x_geomean + width, geomean_new * 1.05, f"{geomean_new:.2f}", ha="center", va="bottom", fontsize=12)
plt.tight_layout()
plt.savefig("speedup_bar_chart_e2e.pdf")
plt.show()
