import pandas as pd
import matplotlib.pyplot as plt

# === Input CSV ===
csv_file = "mapping_sensitivity.csv"  # change to your CSV file name
df = pd.read_csv(csv_file)

# === Compute normalized latency and performance ===
df["norm_latency"] = df["latency"] / df["latency"].min()
df["norm_perf"] = 1 / df["norm_latency"]  # higher is better

# === Get top 30 tile mappings ===
top_tiles = (
    df.sort_values("latency")
      .head(30)["tile_mapping"]
      .unique()
)
all_arrs = df["arr_mapping"].unique()

# === Filter dataframe to show all arr_mappings but only top tile_mappings ===
df_plot = df[df["tile_mapping"].isin(top_tiles)]

# === Scatter plot ===
plt.figure(figsize=(3.2, 2.4))  # fits single-column paper figure

sizes = df_plot["norm_perf"] * 200  # scale dot size
plt.scatter(
    df_plot["tile_mapping"],
    df_plot["arr_mapping"],
    s=sizes,
    c="#7fc97f",
    alpha=0.6,
    edgecolor="black",
    linewidth=0.4,
)

# Add text annotation showing max/min ratio
max_perf = df_plot["norm_perf"].max()
min_perf = df_plot["norm_perf"].min()
ratio = max_perf / min_perf
plt.text(0.98, 0.98, f"Max/Min Ratio: {ratio:.2f}x", 
         transform=plt.gca().transAxes, 
         verticalalignment="top", horizontalalignment="right",
         bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
         fontsize=6)

# === Aesthetics ===
plt.xticks(rotation=90, fontsize=6)
plt.yticks(all_arrs, fontsize=6)  # ensure all arr_mapping ticks are shown
plt.xlabel("Tile Mapping", fontsize=8)
plt.ylabel("Array Mapping", fontsize=8)
plt.tight_layout(pad=0.2)

# Optional: highlight best mapping
best = df.loc[df["latency"].idxmin()]
plt.scatter(
    best["tile_mapping"], best["arr_mapping"],
    s=(1 / best["norm_latency"]) * 300,
    c="red", alpha=0.8, edgecolor="black", linewidth=0.5, zorder=3,
)

# === Save ===
# plt.savefig("top30_tile_all_array_scatter.pdf", bbox_inches="tight")
plt.savefig("sensitivity_mapping", dpi=600, bbox_inches="tight")

plt.show()
