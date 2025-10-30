import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def main():
    parser = argparse.ArgumentParser(
        description="Plot relative perf/area improvement (Proteus, PIM vs H100)"
    )
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    args = parser.parse_args()

    input_file = args.input
    output_file = os.path.splitext(input_file)[0] + "_speedup_column.pdf"

    # === Load CSV ===
    df = pd.read_csv(input_file)
    h100 = df.iloc[0].values
    pim = df.iloc[2].values
    proteus = df.iloc[1].values
    workloads = df.columns.tolist()

    # === Compute relative improvements ===
    improv_pim = pim / h100
    improv_proteus = proteus / h100
    improv_pim_against_proteus = pim / proteus
    #calculate geomean of improv_pim_against_proteus
    geo_mean_improv = np.exp(np.mean(np.log(improv_pim_against_proteus)))
    print(f"Geometric mean of PIM vs Proteus perf/area improvement: {geo_mean_improv:.2f}×")
    geo_mean_improv_gpu = np.exp(np.mean(np.log(improv_pim)))
    print(f"Geometric mean of PIM vs H100 perf/area improvement: {geo_mean_improv_gpu:.2f}×")

    # === Plot grouped bars ===
    plt.figure(figsize=(6, 5))

    x = np.arange(len(workloads))
    width = 0.25  # width of each bar

    bars1 = plt.bar(x - width, np.ones_like(h100), width, label="H100", color="lightgray", edgecolor="black")
    bars2 = plt.bar(x, improv_proteus, width, label="Proteus", color="#4dac26", edgecolor="black")
    bars3 = plt.bar(x + width, improv_pim, width, label="ReAP(Ours)", color="#8073ac", edgecolor="black")

    # Labels and style
    plt.xticks(x, workloads, rotation=45, ha="right", fontsize=9)
    plt.yscale("log")
    plt.ylabel("Perf/Area vs. H100 (×)", fontsize=9)
    plt.title("Performance per Area Comparison", fontsize=9)
    plt.legend(fontsize=8, frameon=False)

    # Add annotations
    for bars in [bars3]:
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2, 
                height * 1.05, 
                f"{height:.2f}", 
                ha="center", va="bottom", fontsize=7
            )

    plt.tight_layout()
    plt.savefig(f"{output_file}")
    plt.show()

if __name__ == "__main__":
    main()
