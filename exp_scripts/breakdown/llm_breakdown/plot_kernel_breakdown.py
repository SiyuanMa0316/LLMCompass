import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description="Plot latency breakdown bar chart")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to the CSV file")
    parser.add_argument("--output", type=str, default="latency_breakdown.png",
                        help="Output figure name")
    args = parser.parse_args()

    # Load CSV
    df = pd.read_csv(args.input)

    kernels = df["kernel"]
    array_lat = df["array_latency"].values
    red_lat = df["reduction_latency"].values
    io_lat = df["io_overhead"].values

    x = np.arange(len(kernels))
    width = 0.55  # single bar width (stacked)

    plt.figure(figsize=(3.3, 2.0))  # good for ISCA 2-column width

    # Stacked bars
    p1 = plt.bar(x, array_lat, width, label="Array")
    p2 = plt.bar(x, red_lat, width, bottom=array_lat, label="Reduction")
    p3 = plt.bar(x, io_lat, width, bottom=array_lat + red_lat, label="I/O")

    # Labels and ticks
    plt.ylabel("Latency (s)", fontsize=9)
    plt.xticks(x, kernels, rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)

    # Legend (compact for paper)
    plt.legend(fontsize=7, frameon=False)

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {args.output}")

if __name__ == "__main__":
    main()
