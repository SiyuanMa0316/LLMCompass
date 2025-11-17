import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

BASE_FONT = 8
plt.rcParams.update({
    "font.size": BASE_FONT,
    "axes.labelsize": BASE_FONT + 1,
    "axes.titlesize": BASE_FONT,
    "legend.fontsize": BASE_FONT,
    "xtick.labelsize": BASE_FONT,
    "ytick.labelsize": BASE_FONT,
    "figure.dpi": 300,
})

FIG_WIDTH = 4
FIGSIZE = (FIG_WIDTH, 2.7)
ANNOT_FONTSIZE = BASE_FONT - 1
AXHLINE_WIDTH = 0.3
BAR_LINEWIDTH = 0.3
BAR_WIDTH = 0.25
SPINE_LINEWIDTH = 0.8


def geom_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if np.any(values <= 0):
        raise ValueError("Geometric mean requires strictly positive values.")
    return float(np.exp(np.mean(np.log(values))))

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
speedups_h100 = np.ones_like(baseline, dtype=float)
speedups_proteus = baseline / proteus
speedups_new = baseline / new

# if len(workloads) < 10:
#     raise ValueError("Expected at least 10 workloads to compute requested geomeans.")

overall_geomean_label = "Geomean"

workloads_all: list[str] = list(workloads)
speedups_h100_all: list[float] = speedups_h100.tolist()
speedups_proteus_all: list[float] = speedups_proteus.tolist()
speedups_new_all: list[float] = speedups_new.tolist()

speedups_h100_all.append(1.0)
speedups_proteus_all.append(geom_mean(speedups_proteus))
speedups_new_all.append(geom_mean(speedups_new))
workloads_all.append(overall_geomean_label)

speedups_h100_all = np.asarray(speedups_h100_all, dtype=float)
speedups_proteus_all = np.asarray(speedups_proteus_all, dtype=float)
speedups_new_all = np.asarray(speedups_new_all, dtype=float)

# Plot
plt.figure(figsize=FIGSIZE)

x = np.arange(len(workloads_all))
geomean_indices = [i for i, label in enumerate(workloads_all) if "Geomean" in label]
geomean_mask = np.array([idx in geomean_indices for idx in range(len(workloads_all))], dtype=bool)
non_geomean_mask = ~geomean_mask

colors_h100 = np.full(len(workloads_all), "#a19a9a", dtype=object)
colors_proteus = np.full(len(workloads_all), "#d36a6a", dtype=object)
colors_new = np.full(len(workloads_all), "#6b96c2", dtype=object)


bars1 = plt.bar(
    x[non_geomean_mask] - BAR_WIDTH,
    speedups_h100_all[non_geomean_mask],
    BAR_WIDTH,
    label="H100",
    color=colors_h100[non_geomean_mask],
    edgecolor="black",
    linewidth=BAR_LINEWIDTH,
)
bars2 = plt.bar(
    x,
    speedups_proteus_all,
    BAR_WIDTH,
    label="Proteus",
    color=colors_proteus,
    edgecolor="black",
    linewidth=BAR_LINEWIDTH,
)
bars3 = plt.bar(
    x + BAR_WIDTH,
    speedups_new_all,
    BAR_WIDTH,
    label="DREAM(Ours)",
    color=colors_new,
    edgecolor="black",
    linewidth=BAR_LINEWIDTH,
)

# for idx, bar in enumerate(bars2):
#     if geomean_mask[idx]:
#         bar.set_hatch("/")
#         bar.set_edgecolor((0, 0, 0, 0.5))

# for idx, bar in enumerate(bars3):
#     if geomean_mask[idx]:
#         bar.set_hatch("/")
#         bar.set_edgecolor((0, 0, 0, 0.5))

# Labels and style


plt.xticks(x, workloads_all, rotation=30, ha="right", fontsize=plt.rcParams["xtick.labelsize"], color="black")
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(SPINE_LINEWIDTH)
ax.tick_params(axis="x", labelsize=plt.rcParams["xtick.labelsize"])
ax.tick_params(axis="y", labelsize=plt.rcParams["ytick.labelsize"])
for tick in ax.get_xticklabels():
    tick.set_color("black")
plt.yscale("log")
plt.ylabel("Normalized throughput")
# Set asymmetric y-limits: compress lower range (10^-4 to 10^-1), expand upper range (1 to 10^3)

# --- Align both y axes (so 10^0 = 1 line up) ---
ymin = (ax.get_ylim()[0])
ymax = (ax.get_ylim()[1])*15
ax.set_ylim(ymin, ymax)
# ax2.set_ylim(ymin, ymax)
plt.axhline(y=1, color="black", linestyle="--", linewidth=AXHLINE_WIDTH, alpha=0.7)

# Light pastel tick labels for geomeans
# for idx, tick in enumerate(plt.gca().get_xticklabels()):
#     if idx in geomean_indices:
#         tick.set_color("#51a4f6")

# Add annotations for DREAM bars
for idx, (bar, value) in enumerate(zip(bars3, speedups_new_all)):
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height * 1.1,
        f"x{value:.1f}",
        ha="center",
        va="bottom",
        fontsize=ANNOT_FONTSIZE,
        # color="#336fa4" if idx in geomean_indices else "black",
    )

handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(
    handles,
    labels,
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.035),
    ncol=3,
)

plt.tight_layout(pad=0.35)
plt.savefig("speedup_bar_chart.pdf", bbox_inches="tight", pad_inches=0.02)
plt.show()
