from turtle import color
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

# =======================
# === Constants (values unchanged) ===
# =======================
FIGSIZE = (6, 4)          # single-column figure
ANNOT_FONTSIZE = 8
XTICK_FONTSIZE = 5
YLABEL_FONTSIZE = 8
XLABEL_FONTSIZE = 8
YTICK_FONTSIZE = 7
EDGE_COLOR = "black"
EDGE_LINEWIDTH = 0.4
BASELINE_COLOR = "gray"
BASELINE_LS = "--"
BASELINE_LW = 0.6

# Existing palette (do not change)
colors = ['#fff7bc', '#fec44f', "#dd7e3e"]

# PE-utilization line style (new constants)
PE_LINE_COLOR = "#336ae0"
PE_LINE_LS = "--"
PE_LINE_LW = 0.8
PE_MARKER = "o"
PE_MARKERSIZE = 2
LATENCY_YLABEL = "Normalized latency"
PE_YLABEL = "Normalized PE utilization"   # right-axis label
WORKLOAD_YLABEL = "Normalized workload capacity"
GROUP_SIZE = 3

WORKLOAD_COLOR = "#01893A"

# Visual upward shift in percentage points (data-space offset, labels corrected via formatter)
PE_BASE_SHIFT = 6.0  # try 6–12 if you need more/less lift

# === Input CSV ===
csv_file = "latencies_run_gemv_output.csv"  # replace with your filename
df = pd.read_csv(csv_file)

# === Parse data ===
workloads = df["workload"].astype(str).str.replace("GEMM_", "", regex=False).values
latencies = df["simulated_latency"].astype(float).values
norm_latencies = latencies / latencies.min()

workloads_int = np.array([int(w.split('x')[0]) * int(w.split('x')[1]) * int(w.split('x')[2]) for w in workloads])
print(workloads, workloads_int.min())
workloads_normalized = workloads_int/workloads_int.min()

# PE utilization -> percent for right axis
# pe_utils_pct = (df["pe_utilization"].astype(float).values) * 100.0
pe_utils = df["pe_utilization"].astype(float).values
pe_utils_pct = pe_utils / pe_utils[0]

# === Bar colors (unchanged logic) ===
bar_colors = [colors[i // 3] for i in range(len(workloads))]

# === Plot ===
fig, ax = plt.subplots(figsize=FIGSIZE)
x = np.arange(len(workloads))

# Make bars wider
BAR_WIDTH = 0.75  # wider bars (default is ~0.6)
bars = ax.bar(
    x,
    norm_latencies,
    color=bar_colors,
    edgecolor=EDGE_COLOR,
    linewidth=EDGE_LINEWIDTH,
    width=BAR_WIDTH,
)

# === (Optional) Log scale ===
# ax.set_yscale("log")

# === Annotation placement logic (inside bars) ===
ylim_top = max(norm_latencies) * 1.2
ax.set_ylim(0, ylim_top)

for i, val in enumerate(norm_latencies):
    # Place text **inside** the bar (middle)
    y_pos = val * 0.5
    ax.text(
        i,
        y_pos,
        f"x{val:.1f}",
        ha="center",
        va="center",
        fontsize=ANNOT_FONTSIZE,
        # rotation=90,
        color="black",
    )

# === Right axis: PE utilization (%) dashed line with visual upward shift via data offset ===
ax2 = ax.twinx()
ax2.set_ylabel(PE_YLABEL, fontsize=YLABEL_FONTSIZE)

# Plot shifted data so the curve sits higher, but format ticks to show true values
pe_plot_vals = pe_utils_pct + PE_BASE_SHIFT

# Reasonable y-limits for the shifted data
if len(pe_plot_vals) > 0:
    ymax = float(np.nanmax(pe_plot_vals) * 1.2)
    ax2.set_ylim(0, max(5.0 + PE_BASE_SHIFT, ymax))

# Plot the line (shifted)
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
    # Place text **inside** the bar (middle)
    y_pos = val + 8
    ax2.text(
        i,
        y_pos,
        f"x{val:.1f}",
        ha="center",
        va="bottom",
        fontsize=ANNOT_FONTSIZE,
        # rotation=90,
        color=PE_LINE_COLOR,
    )

# Show true % values by subtracting the shift in the formatter
ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{max(y - PE_BASE_SHIFT, 0):.0f}"))

# Color the right axis
ax2.spines['right'].set_color(PE_LINE_COLOR)
ax2.yaxis.label.set_color(PE_LINE_COLOR)
ax2.tick_params(axis='y', labelcolor=PE_LINE_COLOR, labelsize=YTICK_FONTSIZE)

# === Aesthetics (left axis unchanged) ===
ax.set_xticks(x)
ax.set_xticklabels(workloads, rotation=90, ha="center", fontsize=XTICK_FONTSIZE)
ax.set_ylabel(LATENCY_YLABEL, fontsize=YLABEL_FONTSIZE)
ax.set_xlabel("Workload (MxKxN)", fontsize=XLABEL_FONTSIZE)
ax.tick_params(axis="y", labelsize=YTICK_FONTSIZE)

# Right axis 3: workloads_normalized line with offset to the right
ax3 = ax.twinx()
ax3.spines['right'].set_position(('outward', 30))
ax3.set_ylabel(WORKLOAD_YLABEL, fontsize=YLABEL_FONTSIZE, color=WORKLOAD_COLOR)

workload_plot_vals = workloads_normalized + PE_BASE_SHIFT
ax3.set_ylim(0, max(5.0 + PE_BASE_SHIFT, float(np.nanmax(workload_plot_vals) * 1.2)))

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
    # Place text **inside** the bar (middle)
    y_pos = val
    if i < 5:
        y_pos += 20
    if i == 6:
        y_pos += 10
    else:
        y_pos -= 10
    if i == 8:
        i += +0.3
    ax3.text(
        i,
        y_pos,
        f"x{int(val)}",
        ha="center",
        va="bottom",
        fontsize=ANNOT_FONTSIZE,
        # rotation=90,
        color=WORKLOAD_COLOR,
    )


# ax3.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{max(y - PE_BASE_SHIFT, 0):.0f}"))
ax3.spines['right'].set_color(WORKLOAD_COLOR)
ax3.yaxis.label.set_color(WORKLOAD_COLOR)
ax3.tick_params(axis='y', labelcolor=WORKLOAD_COLOR, labelsize=YTICK_FONTSIZE)

# ax.axhline(1.0, color=BASELINE_COLOR, linestyle=BASELINE_LS, linewidth=BASELINE_LW)
fig.tight_layout(pad=0.2)
ax.grid(True, axis='y', alpha=0.6, linestyle='--', linewidth=0.5, color="#37322F")
# === Legends (keep only PE line as in your latest version) ===
legend_elements = [
    # Patch(facecolor=colors[0], edgecolor=EDGE_COLOR, linewidth=EDGE_LINEWIDTH,
    #    label=LATENCY_YLABEL),
    Line2D([0], [0], color=PE_LINE_COLOR, linestyle=PE_LINE_LS, linewidth=PE_LINE_LW,
        marker=PE_MARKER, markersize=PE_MARKERSIZE, label=PE_YLABEL),
    Line2D([0], [0], color=WORKLOAD_COLOR, linestyle=PE_LINE_LS, linewidth=PE_LINE_LW,
        marker=PE_MARKER, markersize=PE_MARKERSIZE, label=WORKLOAD_YLABEL),
]
ax.legend(handles=legend_elements, fontsize=ANNOT_FONTSIZE, loc='upper left')

# === Save ===
# plt.savefig("gemm_latency_bar_log_shades_adaptive.pdf", bbox_inches="tight")
plt.savefig("sensitivity_gemv_size.png", dpi=600, bbox_inches="tight")

plt.show()
