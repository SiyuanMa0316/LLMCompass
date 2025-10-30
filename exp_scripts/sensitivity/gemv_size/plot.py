import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === Input CSV ===
csv_file = "latencies_run_gemv_output.csv"  # replace with your filename
df = pd.read_csv(csv_file, header=None)

# === Parse data ===
workloads = df.iloc[0].values
latencies = df.iloc[1].astype(float).values

# === Clean workload names (remove 'GEMM_') ===
workloads = [w.replace("GEMM_", "") for w in workloads]

# === Normalize latencies ===
norm_latencies = latencies / latencies[0]

# === Define colors (light → medium → dark) ===
colors = ['#fff7bc', '#fec44f', '#d95f0e']
bar_colors = [colors[i // 3] for i in range(len(workloads))]

# === Plot ===
plt.figure(figsize=(1.6, 2.2))  # single-column figure
bars = plt.bar(
    range(len(workloads)),
    norm_latencies,
    color=bar_colors,
    edgecolor="black",
    linewidth=0.4,
)

# === Log scale ===
# plt.yscale("log")

# === Annotation placement logic ===
ylim_top = max(norm_latencies) * 1.2  # estimated upper space
plt.ylim(1e0 * 0.9, ylim_top)         # ensure consistent view range

n = len(norm_latencies)
for i, val in enumerate(norm_latencies):
    # determine position
    y_pos = val
    # if i >= n - 3:  # if near top, place inside
    #     y_pos = val / 1.2
    #     va = "top"
    # else:
    va = "bottom"

    plt.text(
        i,
        y_pos,
        f" ×{val:.1f}",
        ha="center",
        va=va,
        fontsize=6,
        rotation=90,
    )

# === Aesthetics ===
plt.xticks(range(len(workloads)), workloads, rotation=90, ha="center", fontsize=5)
plt.ylabel("Normalized Latency", fontsize=8)
plt.xlabel("Workload (M×K×N)", fontsize=8)
plt.yticks(fontsize=7)
plt.axhline(1.0, color="gray", linestyle="--", linewidth=0.6)
plt.tight_layout(pad=0.2)

# === Save ===
# plt.savefig("gemm_latency_bar_log_shades_adaptive.pdf", bbox_inches="tight")
plt.savefig("sensitivity_gemv_size.png", dpi=600, bbox_inches="tight")

plt.show()
