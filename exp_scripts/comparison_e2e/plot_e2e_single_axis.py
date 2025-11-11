import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Half-column plot: unified speedup (decode throughput + prefill)")
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
plt.figure(figsize=(2.3, 0.8))  # half-column figure
x = np.arange(len(workloads))
width = 0.25

ax = plt.gca()

# --- Bars: decode throughput speedups ---
ax.bar(x - width, np.ones_like(baseline_thr), width, label="H100",
       color="#b8b0b0", edgecolor=None)
ax.bar(x, thr_speedup_proteus, width, label="Token TP(Proteus)",
       color="#ef7b7b", edgecolor=None)
bars_new = ax.bar(x + width, thr_speedup_new, width, label="Token TP(DREAM)",
                  color="#82b4e7", edgecolor=None)

ax.set_yscale("log")
ax.set_ylabel("Speedup vs H100", fontsize=7)
ax.axhline(1, color="black", linestyle="--", linewidth=0.6)

# --- Line plot: prefill speedups (on same y-axis) ---
ax.plot(x, prefill_speedup_proteus, color="#c75d5d", marker="o", markersize=4.0,
        linewidth=1.3, label="Prefill TP(Proteus)")
ax.plot(x + width, prefill_speedup_new, color="#3f7ec5", marker="s", markersize=4.0,
        linewidth=1.3, label="Prefill TP(DREAM)")

ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] * 8)

# --- Geomean ---
geo_thr_proteus = np.exp(np.mean(np.log(thr_speedup_proteus)))
geo_thr_new = np.exp(np.mean(np.log(thr_speedup_new)))
geo_pref_proteus = np.exp(np.mean(np.log(prefill_speedup_proteus)))
geo_pref_new = np.exp(np.mean(np.log(prefill_speedup_new)))

x_geo = len(workloads)
ax.bar(x_geo - width, 1, width, color="#b8b0b0", edgecolor=None)
ax.bar(x_geo, geo_thr_proteus, width, color="#ef7b7b", edgecolor=None)
bar_geo_new = ax.bar(x_geo + width, geo_thr_new, width, color="#82b4e7", edgecolor=None)
ax.plot([x_geo], [geo_pref_proteus], marker="o", color="#c75d5d", markersize=4.0)
ax.plot([x_geo + width], [geo_pref_new], marker="s", color="#3f7ec5", markersize=4.0)

# --- Numeric labels for ReMAP (Ours) bars ---
for bar in list(bars_new) + list(bar_geo_new):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height * 1.05, f"{height:.1f}",
            ha="center", va="bottom", fontsize=4, clip_on=False)

# --- Numeric labels for Prefill (ReMAP) line ---
for xi, val in zip(list(x + width) + [x_geo + width],
                   list(prefill_speedup_new) + [geo_pref_new]):
    ax.text(xi, val * 4, f"{val:.1f}",
            fontsize=4, ha="center", va="bottom", clip_on=False)

# --- Styling ---
plt.xticks(list(x) + [x_geo], workloads + ["Geomean"], rotation=25, ha="right", fontsize=5)
ax.tick_params(axis="y", labelsize=5)

# Legends merged (single axis)
ax.legend(loc="upper left", fontsize=5, frameon=False, handlelength=1.2, ncol=2, bbox_to_anchor=(0.0, 1.6))

# plt.title("Throughput and TTFT Speedup", fontsize=7, y=1.18)
# plt.tight_layout(pad=0.25)
plt.savefig("speedup.pdf", bbox_inches="tight")
plt.show()
