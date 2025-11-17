import numpy as np
import matplotlib.pyplot as plt
import sys
import csv

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
FIGSIZE = (FIG_WIDTH, 3.2)
BAR_LINEWIDTH = 0.3
BAR_WIDTH = 0.25
AXHLINE_WIDTH = 0.3
SPINE_LINEWIDTH = 0.8

def coerce_float_row(row):
    vals = []
    for x in row:
        s = x.strip()
        if s == '':
            continue
        vals.append(float(s))
    return vals

if len(sys.argv) != 2:
    print("Usage: python compact_relative_perf_vs_pim_legend_top.py <data.csv>")
    sys.exit(1)

csv_file = sys.argv[1]

with open(csv_file, 'r', newline='') as f:
    reader = csv.reader(f)
    raw_rows = [[c.strip() for c in r if c.strip() != ''] for r in reader]

rows = [r for r in raw_rows if len(r) > 0]
if len(rows) < 6:
    raise ValueError("Expected at least 6 rows: labels + H100 + PIM + 3 ablations.")

labels = rows[0]
data_rows = [coerce_float_row(r) for r in rows[1:6]]

# Validate consistency
num_cols_set = {len(r) for r in data_rows}
if len(num_cols_set) != 1:
    raise ValueError(f"Numeric rows have inconsistent lengths: {sorted(num_cols_set)}.")
num_cols = list(num_cols_set)[0]

# Handle repeated label blocks
if len(labels) != num_cols:
    if len(labels) > num_cols and len(labels) % num_cols == 0:
        labels = labels[:num_cols]
    else:
        raise ValueError(f"Label count ({len(labels)}) != data columns ({num_cols}).")

data = np.array(data_rows, dtype=float)
h100, pim, no_subarray_broadcast, no_popcount_reduction, no_locality_buffer = data

# Compute relative performance vs PIM (baseline = 1.0)
rel = {
    "Complete Configuration": np.ones_like(pim),
    "PR Removed ": pim / no_subarray_broadcast,
    "PR & BU Removed": pim / no_popcount_reduction,
    "PR & BU & LB Removed": pim / no_locality_buffer,
}

fig, ax = plt.subplots(figsize=FIGSIZE)
series_names = list(rel.keys())
series_values = [rel[name] for name in series_names]
n_series = len(series_names)
offsets = (np.arange(n_series) - (n_series - 1) / 2.0) * BAR_WIDTH

# Append geometric-mean column per series to create an aggregate bar
geo_label = "Geomean"
geo_values = []
for values in series_values:
    if np.any(values <= 0):
        raise ValueError("Geometric mean is undefined for non-positive performance values.")
    geo_values.append(np.exp(np.mean(np.log(values))))

series_values = [np.append(values, geo) for values, geo in zip(series_values, geo_values)]
labels = labels + [geo_label]

x = np.arange(len(labels))

# Draw bars
colors = ["#EEB78F", "#E00F47", "#20B2AA", "#9932CC"]
for i, (name, values) in enumerate(zip(series_names, series_values)):
    ax.bar(
        x + offsets[i],
        values,
        width=BAR_WIDTH,
        label=name,
        linewidth=BAR_LINEWIDTH,
        edgecolor="black",
        color=colors[i],
    )


# Labels and grid
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha='right')
ax.set_ylabel("Normalized performance")
# ax.set_title("Ablation Study of Added Peripheral Components", pad=22)
ax.grid(True, axis='y', linestyle='--', linewidth=0.3, alpha=0.7, color="#272329")
ax.axhline(1.0, color="black", linestyle="--", linewidth=AXHLINE_WIDTH, alpha=0.7)
for spine in ax.spines.values():
    spine.set_linewidth(SPINE_LINEWIDTH)
ax.tick_params(axis='x', labelsize=plt.rcParams['xtick.labelsize'])
ax.tick_params(axis='y', labelsize=plt.rcParams['ytick.labelsize'])
ymin = (ax.get_ylim()[0])
ymax = (ax.get_ylim()[1])*5
ax.set_ylim(ymin, ymax)
# Legend: placed below title, above the plot, with extra spacing
legend = ax.legend(
    frameon=False,
    ncol=2,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.4),
    handlelength=1.2,
    columnspacing=0.8,
)

caption = r"$\mathbf{PR}$ Popcount Reduction    $\mathbf{BU}$ Broadcasting Unit    $\mathbf{LB}$ Locality Buffer"
fig.text(0.52, 0.74, caption, ha='center', va='top', fontsize=BASE_FONT-1)

plt.tight_layout(pad=0.35)
plt.savefig("ablation_accumulative.png", bbox_inches="tight", pad_inches=0.02)
plt.show()
