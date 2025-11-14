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
if len(rows) < 7:
    raise ValueError("Expected at least 6 rows: labels + H100 + PIM + 3 ablations.")

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
h100, pim_base, pim_int4, pim_256g, pim_64g, pim_8g = data

# Compute relative performance vs PIM (baseline = 1.0)
rel = {
    "1024GB": np.ones_like(pim_base),
    # "int4": pim_base / pim_int4,
    "256GB": pim_base / pim_256g,
    "64GB": pim_base / pim_64g,
    "8GB": pim_base / pim_8g,
    
}

# --- Compact academic style ---
FIGSIZE = (3.4, 2.5)
BASE_FONT = 7
TITLE_FONT = 8
LEGEND_FONT = 6
TICK_FONT = 6

plt.rcParams.update({
    "font.size": BASE_FONT,
    "axes.labelsize": BASE_FONT,
    "axes.titlesize": TITLE_FONT,
    "legend.fontsize": LEGEND_FONT,
    "xtick.labelsize": TICK_FONT,
    "ytick.labelsize": TICK_FONT,
    "figure.dpi": 300,
})

fig, ax = plt.subplots(figsize=FIGSIZE)
for spine in ax.spines.values():
    spine.set_linewidth(0.8)

x = np.arange(len(labels))
series_names = list(rel.keys())
series_values = [rel[name] for name in series_names]
n_series = len(series_names)
bar_width = 0.18
offsets = (np.arange(n_series) - (n_series - 1) / 2.0) * bar_width

# Draw bars
for i, (name, values) in enumerate(zip(series_names, series_values)):
    colors = ["#4E8FAB",  "#E7A138", "#1E9012","#D04B92"]
    color = colors[i % len(colors)]
    ax.bar(x + offsets[i], values, width=bar_width, label=name, color=color, linewidth=0.3, edgecolor="black")

    # Add horizontal lines for memory capacity thresholds
    capacities = [256, 64, 8]
    for cap in capacities:
        ax.axhline(y=cap/1024, color="#444244", linestyle='--', linewidth=0.3, alpha=0.5)
        ax.text(len(labels)+0.1, cap/1024, f'{cap}GB', fontsize=5, va='bottom', color='black')

# Labels and grid
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=TICK_FONT)
ax.set_ylabel("Normalized throughput", fontsize=BASE_FONT)
ax.set_title("Sensitivity Study: Memory Capacity", pad=12, fontsize=TITLE_FONT)
ax.set_yscale("log")
# ax.grid(True, axis='y', linestyle='--', linewidth=0.3, alpha=0.6, color="#444244")
legend = ax.legend(
    frameon=False,
    ncol=5,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.13),
    handlelength=1.2,
    columnspacing=0.8,
    fontsize=LEGEND_FONT,
)
plt.tight_layout(pad=0.6)
plt.subplots_adjust(top=0.78)

plt.savefig("sensitivity_capacity.png", bbox_inches='tight')
plt.show()
