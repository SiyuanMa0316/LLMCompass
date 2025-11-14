import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from matplotlib.patches import Patch

FIGSIZE = (10, 8)
XTICK_FONTSIZE = 13
YLABEL_FONTSIZE = 18
LEGEND_FONTSIZE = 14
ANNOT_FONTSIZE = 12
AXHLINE_WIDTH = 1.0


def geom_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if np.any(values <= 0):
        raise ValueError("Geometric mean requires strictly positive values.")
    return float(np.exp(np.mean(np.log(values))))

def palette(color_main: str, color_geomean: str) -> list[str]:
    # Build a per-workload color list: use color_geomean for Geomean labels, otherwise color_main.
    # This function expects workloads_all to be defined before it is called (the plotting code
    # constructs workloads_all earlier).
    try:
        return [color_geomean if "Geomean" in lbl else color_main for lbl in workloads_all]
    except NameError:
        # If called before workloads_all is available, return a minimal two-color list to avoid failure.
        return [color_main, color_geomean]


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

first_group = slice(0, len(workloads)//2)
last_group = slice(len(workloads)//2, len(workloads))

extra_labels = [
    "Prefill_Geomean",
    "Decode_Geomean",
    "Overall_Geomean",
]

extra_h100 = np.ones(len(extra_labels), dtype=float)
extra_proteus = np.array([
    geom_mean(speedups_proteus[first_group]),
    geom_mean(speedups_proteus[last_group]),
    geom_mean(speedups_proteus),
])
extra_new = np.array([
    geom_mean(speedups_new[first_group]),
    geom_mean(speedups_new[last_group]),
    geom_mean(speedups_new),
])

first_group_end = first_group.stop
last_group_start = last_group.start

workloads_all: list[str] = []
speedups_h100_all: list[float] = []
speedups_proteus_all: list[float] = []
speedups_new_all: list[float] = []

def extend_segment(start: int, end: int) -> None:
    workloads_all.extend(workloads[start:end])
    speedups_h100_all.extend(speedups_h100[start:end])
    speedups_proteus_all.extend(speedups_proteus[start:end])
    speedups_new_all.extend(speedups_new[start:end])

extend_segment(0, first_group_end)
workloads_all.append(extra_labels[0])
speedups_h100_all.append(extra_h100[0])
speedups_proteus_all.append(extra_proteus[0])
speedups_new_all.append(extra_new[0])

if last_group_start > first_group_end:
    extend_segment(first_group_end, last_group_start)

extend_segment(last_group_start, len(workloads))
workloads_all.append(extra_labels[1])
speedups_h100_all.append(extra_h100[1])
speedups_proteus_all.append(extra_proteus[1])
speedups_new_all.append(extra_new[1])

workloads_all.append(extra_labels[2])
speedups_h100_all.append(extra_h100[2])
speedups_proteus_all.append(extra_proteus[2])
speedups_new_all.append(extra_new[2])

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

width = 0.25

bars1 = plt.bar(
    x[non_geomean_mask] - width,
    speedups_h100_all[non_geomean_mask],
    width,
    label="H100",
    color=colors_h100[non_geomean_mask],
    edgecolor="black",
)
bars2 = plt.bar(x, speedups_proteus_all, width, label="Proteus", color=colors_proteus, edgecolor="black")
bars3 = plt.bar(x + width, speedups_new_all, width, label="DREAM(Ours)", color=colors_new, edgecolor="black")

for idx, bar in enumerate(bars2):
    if geomean_mask[idx]:
        bar.set_hatch("/")
        bar.set_edgecolor((0, 0, 0, 0.5))

for idx, bar in enumerate(bars3):
    if geomean_mask[idx]:
        bar.set_hatch("/")
        bar.set_edgecolor((0, 0, 0, 0.5))

# Labels and style
plt.xticks(x, workloads_all, rotation=15, ha="right", fontsize=XTICK_FONTSIZE, color="black")
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(1.5)
ax.tick_params(axis="both", labelsize=XTICK_FONTSIZE)
for tick in ax.get_xticklabels():
    tick.set_color("black")
plt.yscale("log")
plt.ylabel("Normalized throughput", fontsize=YLABEL_FONTSIZE)
plt.axhline(y=1, color="black", linestyle="--", linewidth=AXHLINE_WIDTH, alpha=0.7)
plt.legend(fontsize=LEGEND_FONTSIZE, frameon=False)

# Light pastel tick labels for geomeans
# for idx, tick in enumerate(plt.gca().get_xticklabels()):
#     if idx in geomean_indices:
#         tick.set_color("#51a4f6")

# Add annotations for DREAM bars
for idx, (bar, value) in enumerate(zip(bars3, speedups_new_all)):
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height * 1.05,
        f"x{value:.1f}",
        ha="center",
        va="bottom",
        fontsize=ANNOT_FONTSIZE,
        # color="#336fa4" if idx in geomean_indices else "black",
    )

handles, labels = plt.gca().get_legend_handles_labels()
if geomean_indices:
    proteus_geo_patch = Patch(
        facecolor=colors_proteus[geomean_indices[0]],
        edgecolor=(0, 0, 0, 0.5),
        hatch="/",
        label="Proteus Geomean",
    )
    dream_geo_patch = Patch(
        facecolor=colors_new[geomean_indices[0]],
        edgecolor=(0, 0, 0, 0.5),
        hatch="/",
        label="DREAM Geomean",
    )
    handles = list(handles) + [proteus_geo_patch, dream_geo_patch]
    labels = list(labels) + ["Proteus Geomean", "DREAM Geomean"]
plt.legend(handles, labels, fontsize=LEGEND_FONTSIZE, frameon=False, loc="upper left")

plt.tight_layout()
plt.savefig("speedup_bar_chart.pdf")
plt.show()
