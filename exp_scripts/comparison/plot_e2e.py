import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Half-column plot: decode throughput + prefill speedup (vs H100)")
parser.add_argument("--throughput", type=str, default="throughput_decode_1024to128_run_all_output_11-8.csv",
                    help="Path to throughput CSV (tokens/s)")
parser.add_argument("--prefill", type=str, default="latencies_prefill_1024to128_run_all_output_11-8.csv",
                    help="Path to prefill latency CSV (seconds)")
args = parser.parse_args()

# === Load data ===
df_thr = pd.read_csv(args.throughput)
df_pref = pd.read_csv(args.prefill)

workloads = df_thr.columns.tolist()
baseline_thr = df_thr.iloc[0].values
proteus_thr = df_thr.iloc[1].values
new_thr = df_thr.iloc[2].values

baseline_pref = df_pref.iloc[0].values
proteus_pref = df_pref.iloc[1].values
new_pref = df_pref.iloc[2].values

# === Compute speedups (× vs H100) ===
thr_speedup_proteus = proteus_thr / baseline_thr
thr_speedup_new = new_thr / baseline_thr
prefill_speedup_proteus = baseline_pref / proteus_pref
prefill_speedup_new = baseline_pref / new_pref

# === Plot ===
plt.figure(figsize=(2.3, 2.4))  # slightly taller to prevent label overlap
x = np.arange(len(workloads))
width = 0.25

ax = plt.gca()

# --- Bars: decode throughput speedups ---
ax.bar(x - width, np.ones_like(baseline_thr), width, label="H100",
       color="#b8b0b0", edgecolor=None, linewidth=0.6)
ax.bar(x, thr_speedup_proteus, width, label="Throughput(Proteus)",
       color="#ef7b7b", edgecolor=None, linewidth=0.6)
bars_new = ax.bar(x + width, thr_speedup_new, width, label="Throughput(DREAM)",
                  color="#82b4e7", edgecolor=None, linewidth=0.6)

ax.set_yscale("log")
ax.set_ylabel("Throughput Speedup", fontsize=5)
ax.axhline(1, color="black", linestyle="--", linewidth=0.6)

# --- Line plot: prefill speedups ---
ax2 = ax.twinx()
ax2.plot(x, prefill_speedup_proteus, color="#c75d5d", marker="o", markersize=3.5,
         linewidth=1.0, label="TTFT(Proteus)")
ax2.plot(x + width, prefill_speedup_new, color="#3f7ec5", marker="s", markersize=3.5,
         linewidth=1.0, label="TTFT(DREAM)")

    

ax2.set_yscale("log")
ax2.set_ylabel("Prefill Speedup", fontsize=5)

# --- Align both y axes (so 10^0 = 1 line up) ---
ymin = min(ax.get_ylim()[0], ax2.get_ylim()[0])
ymax = max(ax.get_ylim()[1], ax2.get_ylim()[1])*5
ax.set_ylim(ymin, ymax)
ax2.set_ylim(ymin, ymax)

# --- Geomean ---
geo_thr_proteus = np.exp(np.mean(np.log(thr_speedup_proteus)))
geo_thr_new = np.exp(np.mean(np.log(thr_speedup_new)))
geo_pref_proteus = np.exp(np.mean(np.log(prefill_speedup_proteus)))
geo_pref_new = np.exp(np.mean(np.log(prefill_speedup_new)))

x_geo = len(workloads)
ax.bar(x_geo - width, 1, width, color="#b8b0b0", edgecolor=None, linewidth=0.6)
ax.bar(x_geo, geo_thr_proteus, width, color="#ef7b7b", edgecolor=None, linewidth=0.6)
bar_geo_new = ax.bar(x_geo + width, geo_thr_new, width, color="#82b4e7", edgecolor=None, linewidth=0.6)
ax2.plot([x_geo], [geo_pref_proteus], marker="o", color="#c75d5d", markersize=3.5)
ax2.plot([x_geo + width], [geo_pref_new], marker="s", color="#3f7ec5", markersize=3.5)

# --- Numeric labels for ReMAP (Ours) ---
for bar in list(bars_new) + list(bar_geo_new):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height * 1.05, f"{height:.1f}",
            ha="center", va="bottom", fontsize=5, clip_on=False)
    
for xi, val in zip(list(x + width) + [x_geo + width],
                   list(prefill_speedup_new) + [geo_pref_new]):
    ax2.text(xi, val * 2, f"{val:.1f}",
             fontsize=5, ha="center", va="bottom", clip_on=False)

# --- Styling ---
plt.xticks(list(x) + [x_geo], workloads + ["Geomean"], rotation=25, ha="right", fontsize=5)
ax.tick_params(axis="y", labelsize=5)
ax2.tick_params(axis="y", labelsize=5)

# Move legends slightly lower inside plot to avoid overlap
ax.legend(loc="upper left", fontsize=5, frameon=False, handlelength=1.2, bbox_to_anchor=(0.0, 1.25))
ax2.legend(loc="upper right", fontsize=5, frameon=False, handlelength=1.2, bbox_to_anchor=(1.0, 1.25))

# Ensure x-tick rotation applies correctly
for label in ax.get_xticklabels():
    label.set_rotation(25)
    label.set_fontsize(5)
    label.set_ha("right")

plt.title("Decode Throughput and Prefill Speedup", fontsize=5, y=1.2)
plt.tight_layout(pad=0.25)
plt.savefig("throughput_prefill_speedup_halfcol.pdf", bbox_inches="tight")
plt.show()
