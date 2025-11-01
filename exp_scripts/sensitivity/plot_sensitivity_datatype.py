import numpy as np
import matplotlib.pyplot as plt
import sys
import csv

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
# if len(rows) < 7:
#     raise ValueError("Expected at least 6 rows: labels + H100 + PIM + 3 ablations.")

labels = rows[0]
data_rows = [coerce_float_row(r) for r in rows[1:7]]

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
    # "int4": pim_base / pim_int4,
    "int4": pim_base / pim_int4,
    "int2": pim_base / pim_int2,
    
}

# --- Compact academic style ---
plt.rcParams.update({
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "legend.fontsize": 6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "figure.dpi": 300,
})

fig_width = 3.4   # one-column width
fig_height = 2.5  # slightly taller for clarity
fig, ax = plt.subplots(figsize=(fig_width, fig_height))

x = np.arange(len(labels))
series_names = list(rel.keys())
series_values = [rel[name] for name in series_names]
n_series = len(series_names)
bar_width = 0.18
offsets = (np.arange(n_series) - (n_series - 1) / 2.0) * bar_width

# Draw bars
for i, (name, values) in enumerate(zip(series_names, series_values)):
    ax.bar(x + offsets[i], values, width=bar_width, label=name, linewidth=0.3, edgecolor="black")

# Labels and grid
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha='right')
ax.set_ylabel("Normalized performance")
ax.set_title("Sensitivity Study: Data Precision", pad=15, fontsize=8)
ax.grid(True, axis='y', linestyle='--', linewidth=0.3, alpha=0.7, color="#1E1C1F")

# Legend: placed below title, above the plot, with extra spacing
legend = ax.legend(
    frameon=False,
    ncol=5,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.16),  # precisely between title and plot
    handlelength=1.2,
    columnspacing=0.8,
)

# Adjust layout for title–legend spacing
plt.tight_layout(pad=0.5)
plt.subplots_adjust(top=0.78)  # leave ample space for title + legend

plt.savefig("sensitivity_precision.png", bbox_inches='tight')
plt.show()
