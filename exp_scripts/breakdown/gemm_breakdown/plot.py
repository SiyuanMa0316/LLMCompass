import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import math

def main():
    parser = argparse.ArgumentParser(description="Plot ablation latency breakdown (normalized)")
    parser.add_argument("--input", type=str, required=True,
                        help="Input CSV file with ablation results.")
    parser.add_argument("--output", type=str, default="ablation_breakdown.png",
                        help="Output plot filename.")
    args = parser.parse_args()

    # Load CSV
    df = pd.read_csv(args.input)

    # Rename ablations
    rename_map = {
        "base": "Complete",
        "no_popcount": "w/o popcount",
        "no_broadcast": "w/o broadcast",
        "no_buffer": "w/o locality buffer"
    }
    df["ablation"] = df["ablation"].replace(rename_map)

    # Base total latency for normalization
    base_total = df.loc[df["ablation"] == "Complete", "total_latency"].iloc[0]

    ablations = df["ablation"]

    # Breakdown components
    pim_lat = df["array_latency"].values / base_total
    io_lat = (df["reduction_latency"] + df["io_latency"]).values / base_total

    x = np.arange(len(ablations))
    width = 0.55

    plt.figure(figsize=(1.6, 1.5))

    # Stacked bars
    plt.bar(x, pim_lat, width, label="PIM Latency")
    plt.bar(x, io_lat, width, bottom=pim_lat, label="I/O Latency")

    # Axis labels
    plt.ylabel("Normalized Latency", fontsize=5)
    plt.xticks(x, ablations, rotation=45, ha="right", fontsize=5)
    plt.yticks(fontsize=5)

    # ---- Force integer y-ticks and ensure 1 is shown ----
    ymax = max(pim_lat + io_lat)
    ymax = math.ceil(ymax)  # round up to integer

    # Ensure at least 1 is included
    if ymax < 1:
        ymax = 1

    yticks = list(range(0, ymax + 1))
    if 1 not in yticks:
        yticks.append(1)
        yticks = sorted(yticks)

    plt.yticks(yticks)
    plt.ylim(0, ymax)

    # Legend
    plt.legend(fontsize=4, frameon=False)

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {args.output}")

if __name__ == "__main__":
    main()
