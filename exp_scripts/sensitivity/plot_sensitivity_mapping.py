import pandas as pd
import matplotlib.pyplot as plt

# === Input CSV ===
csv_file = "mapping_sensitivity.csv"  # change to your actual file name
df = pd.read_csv(csv_file)

# === Combine tile_mapping and arr_mapping into one label ===
df["mapping_strategy"] = df["tile_mapping"] + " | " + df["arr_mapping"]

# === Normalize latency to best (lowest) latency ===
df["norm_latency"] = df["latency"] / df["latency"].min()

# === Sort and pick top 30 strategies ===
df_sorted = df.sort_values("latency").head(30)

# === Plot ===
plt.figure(figsize=(3.2, 2.2))  # good for one-column figure (IEEE ~3.3in)
bars = plt.bar(
    range(len(df_sorted)),
    df_sorted["norm_latency"],
    color="#7fc97f",
    edgecolor="black",
    linewidth=0.4,
)

# === Aesthetics ===
plt.xticks(
    range(len(df_sorted)),
    df_sorted["mapping_strategy"],
    rotation=90,
    fontsize=6,
)
plt.ylabel("Normalized Latency", fontsize=8)
plt.yticks(fontsize=7)
plt.xlabel("Mapping Strategy", fontsize=8)
plt.tight_layout(pad=0.1)

# Optional: annotate best bar
plt.axhline(1.0, color="gray", linestyle="--", linewidth=0.7)

# === Save ===
plt.savefig("top30_mapping_latency.pdf", bbox_inches="tight")
plt.savefig("top30_mapping_latency.pdf", dpi=600, bbox_inches="tight")

plt.show()
