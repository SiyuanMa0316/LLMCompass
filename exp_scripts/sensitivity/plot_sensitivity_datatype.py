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
FIGSIZE = (FIG_WIDTH, 2.3)
BAR_LINEWIDTH = 0.3
BAR_WIDTH = 0.2
AXHLINE_WIDTH = 0.3
SPINE_LINEWIDTH = 0.8
ANNOT_FONTSIZE = BASE_FONT - 1

def geom_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if np.any(values <= 0):
        raise ValueError("Geometric mean requires strictly positive values.")
    return float(np.exp(np.mean(np.log(values))))

def coerce_float_row(row):
    vals = []
    for x in row:
        s = x.strip()
        if s == '':
            continue
        vals.append(float(s))
    return vals

if len(sys.argv) != 2:
    print("Usage: python plot_sensitivity_datatype.py <data.csv>")
    sys.exit(1)

csv_file = sys.argv[1]

with open(csv_file, 'r', newline='') as f:
    reader = csv.reader(f)
    raw_rows = [[c.strip() for c in r if c.strip() != ''] for r in reader]

rows = [r for r in raw_rows if len(r) > 0]
if len(rows) < 5:
    raise ValueError("Expected at least 5 rows: labels + H100 + PIM variants.")

labels = rows[0]
data_rows = [coerce_float_row(r) for r in rows[1:5]]

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
h100, pim_base, pim_int4, pim_int2 = data

# Compute relative performance vs PIM (baseline = 1.0)
rel = {
    "int8": np.ones_like(pim_base),
    "int4": pim_base / pim_int4,
    "int2": pim_base / pim_int2,
}

# Append geomean
series_names = list(rel.keys())
series_values = [rel[name] for name in series_names]
geomean_label = "Geomean"
geomean_values = [geom_mean(values) for values in series_values]
series_values = [np.append(values, geo) for values, geo in zip(series_values, geomean_values)]
labels = labels + [geomean_label]

fig, ax = plt.subplots(figsize=FIGSIZE)
for spine in ax.spines.values():
    spine.set_linewidth(SPINE_LINEWIDTH)

x = np.arange(len(labels))
n_series = len(series_names)
offsets = (np.arange(n_series) - (n_series - 1) / 2.0) * BAR_WIDTH

# Draw bars
# colors = ["#f8cb7f", "#76da91","#63b2ee"] 
colors = ["#5BB5AC", "#D8B365", "#DE526C"]

for i, (name, values) in enumerate(zip(series_names, series_values)):
    color = colors[i % len(colors)]
    ax.bar(
        x + offsets[i],
        values,
        width=BAR_WIDTH,
        label=name,
        color=color,
        linewidth=BAR_LINEWIDTH,
        edgecolor="black",
        alpha = 0.8
    )

# Labels and grid
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=plt.rcParams["xtick.labelsize"])
ax.set_ylabel("Normalized throughput")
ax.axhline(y=8/8, color="black", linestyle="--", linewidth=AXHLINE_WIDTH, alpha=0.7)
ax.axhline(y=8/4, color="black", linestyle="--", linewidth=AXHLINE_WIDTH, alpha=0.7)
ax.axhline(y=8/2, color="black", linestyle="--", linewidth=AXHLINE_WIDTH, alpha=0.7)
ymin = (ax.get_ylim()[0])
# ymax = (ax.get_ylim()[1])
ymax = 4.5
ax.set_ylim(ymin, ymax)

legend = ax.legend(
    frameon=False,
    ncol=len(colors),
    loc='upper center',
    bbox_to_anchor=(0.5, 1.05),
    handlelength=1.2,
    columnspacing=0.8,
)

plt.tight_layout(pad=0.35)
plt.savefig("sensitivity_precision.png", bbox_inches="tight", pad_inches=0.02)
plt.show()
