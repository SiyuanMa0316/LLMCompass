import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

# =======================
# === Constants matched to GEMV styling ===
# =======================
FIGSIZE = (6, 3)
COMMON_FONT_SIZE = 10
ANNOT_FONTSIZE = COMMON_FONT_SIZE
XTICK_FONTSIZE = COMMON_FONT_SIZE
YLABEL_FONTSIZE = COMMON_FONT_SIZE
XLABEL_FONTSIZE = COMMON_FONT_SIZE
YTICK_FONTSIZE = COMMON_FONT_SIZE
EDGE_COLOR = "black"
EDGE_LINEWIDTH = 0.4
BASELINE_COLOR = "gray"
BASELINE_LS = "--"
BASELINE_LW = 0.6

colors = ['#b2df8a', '#66c2a5', "#44A559"]

PE_LINE_COLOR = "#3a60c1"
PE_LINE_LS = "--"
PE_LINE_LW = 0.8
PE_MARKER = "o"
PE_MARKERSIZE = 2

LATENCY_YLABEL = "Normalized Latency"
PE_YLABEL = "Normalized PE utilization"
WORKLOAD_YLABEL = "Normalized workload capacity"
WORKLOAD_COLOR = "#A73030"

PE_BASE_SHIFT = 6.0
GROUP_SIZE = 3
BAR_WIDTH = 0.75
BAR_GROUP_BASE_SHIFT = 1.3  # lifts annotations for later groups on log axis

# === Input CSV ===
csv_file = "latencies_run_gemm_output.csv"
df = pd.read_csv(csv_file)

# === Parse data ===
workloads = df["workload"].astype(str).str.replace("GEMM_", "", regex=False).values
latencies = df["simulated_latency"].astype(float).values
norm_latencies = latencies / latencies[0]

workloads_int = np.array([int(w.split('x')[0]) * int(w.split('x')[1]) * int(w.split('x')[2]) for w in workloads])
workloads_normalized = workloads_int / workloads_int.min()

pe_utils = df["pe_utilization"].astype(float).values
pe_utils_pct = pe_utils / pe_utils[0]

bar_colors = [colors[i // GROUP_SIZE] for i in range(len(workloads))]

# === Plot ===
fig, ax = plt.subplots(figsize=FIGSIZE)
x = np.arange(len(workloads))

ax.bar(
    x,
    norm_latencies,
    color=bar_colors,
    edgecolor=EDGE_COLOR,
    linewidth=EDGE_LINEWIDTH,
    width=BAR_WIDTH,
)

ax.set_yscale("log")

ylim_top = max(norm_latencies) * 1.2
ax.set_ylim(0.9, ylim_top)

n = len(norm_latencies)
axis_bottom = ax.get_ylim()[0]
for i, val in enumerate(norm_latencies):
    # group_idx = i // GROUP_SIZE
    if i >= 1:
        y_pos = axis_bottom * BAR_GROUP_BASE_SHIFT
        va = "bottom"
    else:
        y_pos = val * 1.05
        if i >= n - GROUP_SIZE:
            y_pos = val / 1.2
            va = "top"
        else:
            va = "bottom"

    ax.text(
        i,
        y_pos,
        f"x{val:.1f}",
        ha="center",
        va=va,
        fontsize=ANNOT_FONTSIZE,
        rotation=90,
    )

# === Right axis: PE utilization ===
ax2 = ax.twinx()
ax2.set_ylabel(PE_YLABEL, fontsize=YLABEL_FONTSIZE)

pe_plot_vals = pe_utils_pct + PE_BASE_SHIFT
if len(pe_plot_vals) > 0:
    ymax = float(np.nanmax(pe_plot_vals) * 1.2)
    ax2.set_ylim(0, max(5.0 + PE_BASE_SHIFT, ymax))

for group_idx, start in enumerate(range(0, len(x), GROUP_SIZE)):
    end = start + GROUP_SIZE
    ax2.plot(
        x[start:end],
        pe_plot_vals[start:end],
        linestyle=PE_LINE_LS,
        linewidth=PE_LINE_LW,
        marker=PE_MARKER,
        markersize=PE_MARKERSIZE,
        color=PE_LINE_COLOR,
        label="PE utilization" if group_idx == 0 else "_nolegend_",
    )

for i, val in enumerate(pe_utils_pct):
    y_pos = val + PE_BASE_SHIFT
    if i == len(pe_utils_pct) - 2:
        y_pos -= 1 
    ax2.text(
        i,
        y_pos,
        f"x{val:.1f}",
        ha="center",
        va="bottom",
        fontsize=ANNOT_FONTSIZE,
        color=PE_LINE_COLOR,
    )

ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{max(y - PE_BASE_SHIFT, 0):.0f}"))
ax2.spines['right'].set_color(PE_LINE_COLOR)
ax2.yaxis.label.set_color(PE_LINE_COLOR)
ax2.tick_params(axis='y', labelcolor=PE_LINE_COLOR, labelsize=YTICK_FONTSIZE)

# === Third axis: normalized workload capacity ===
ax3 = ax.twinx()
ax3.spines['right'].set_position(('outward', 30))
ax3.set_ylabel(WORKLOAD_YLABEL, fontsize=YLABEL_FONTSIZE, color=WORKLOAD_COLOR)

workload_plot_vals = workloads_normalized + PE_BASE_SHIFT
ax3.set_yscale("log")
workload_lower = max(float(np.nanmin(workload_plot_vals) * 0.8), 1e-3)
workload_upper = max(5.0 + PE_BASE_SHIFT, float(np.nanmax(workload_plot_vals) * 1.2))
ax3.set_ylim(workload_lower, workload_upper)

for group_idx, start in enumerate(range(0, len(x), GROUP_SIZE)):
    end = start + GROUP_SIZE
    ax3.plot(
        x[start:end],
        workload_plot_vals[start:end],
        linestyle=PE_LINE_LS,
        linewidth=PE_LINE_LW,
        marker=PE_MARKER,
        markersize=PE_MARKERSIZE,
        color=WORKLOAD_COLOR,
        label="Normalized workload size" if group_idx == 0 else "_nolegend_",
    )

for i, val in enumerate(workloads_normalized):
    y_pos = workload_plot_vals[i]
    if i < GROUP_SIZE + 2:  # keep early points readable
        y_pos += 10
    elif i == len(workloads_normalized) - 1:
        y_pos = val * 0.5
    elif i == len(workloads_normalized) -2:
        y_pos = val * 1.2
    ax3.text(
        i,
        y_pos,
        f"x{int(val)}",
        ha="center",
        va="bottom",
        fontsize=ANNOT_FONTSIZE,
        color=WORKLOAD_COLOR,
    )

ax3.spines['right'].set_color(WORKLOAD_COLOR)
ax3.yaxis.label.set_color(WORKLOAD_COLOR)
ax3.tick_params(axis='y', labelcolor=WORKLOAD_COLOR, labelsize=YTICK_FONTSIZE)

# === Shared aesthetics ===
ax.set_xticks(x)
ax.set_xticklabels(workloads, rotation=15, ha="center", fontsize=XTICK_FONTSIZE - 2)
ax.set_ylabel(LATENCY_YLABEL, fontsize=YLABEL_FONTSIZE)
# ax.set_xlabel("Workload (MxKxN)", fontsize=COMMON_FONT_SIZE)
ax.tick_params(axis="y", labelsize=YTICK_FONTSIZE)
ax.axhline(1.0, color=BASELINE_COLOR, linestyle=BASELINE_LS, linewidth=BASELINE_LW)

fig.tight_layout(pad=0.2)
ax.grid(True, axis='y', alpha=0.6, linestyle='--', linewidth=0.5, color="#37322F")

legend_elements = [
    Line2D([0], [0], color=PE_LINE_COLOR, linestyle=PE_LINE_LS, linewidth=PE_LINE_LW,
           marker=PE_MARKER, markersize=PE_MARKERSIZE, label=PE_YLABEL),
    Line2D([0], [0], color=WORKLOAD_COLOR, linestyle=PE_LINE_LS, linewidth=PE_LINE_LW,
           marker=PE_MARKER, markersize=PE_MARKERSIZE, label=WORKLOAD_YLABEL),
]
ax.legend(handles=legend_elements, fontsize=ANNOT_FONTSIZE, loc='upper left')

# === Save ===
plt.savefig("sensitivity_gemm_size.png", dpi=600, bbox_inches="tight")

plt.show()
