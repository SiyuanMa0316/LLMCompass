import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def main():
    parser = argparse.ArgumentParser(
        description="Plot relative perf/area improvement (Proteus, PIM vs H100)"
    )
    parser.add_argument("--input", required=False, help="Path to input CSV file", default="perf_per_area_run_all_output_10-9.csv")
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
    plt.figure(figsize=(8, 6))

    x = np.arange(len(workloads))
    width = 0.2  # width of each bar

    bars1 = plt.bar(x - width, np.ones_like(h100), width, label="H100", color="#cccccc", edgecolor="black")
    bars2 = plt.bar(x, improv_proteus, width, label="Proteus", color="#ff7f0e", edgecolor="black")
    bars3 = plt.bar(x + width, improv_pim, width, label="ReMAP(Ours)", color="#1fc430", edgecolor="black")

    # Labels and style
    plt.xticks(x, workloads, rotation=30, ha="right", fontsize=12)
    plt.yscale("log")
    plt.ylabel("Perf./ area normalized to H100", fontsize=15)
    # plt.title("Performance per Area Comparison", fontsize=15)
    plt.legend(fontsize=10, frameon=False)

    # Add annotations
    for bars in [bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 1:
                plt.text(
                    bar.get_x() + bar.get_width() * 1.2, 
                    height * 1.05, 
                    f"{height:.2f}", 
                    ha="center", va="bottom", fontsize=12
                )
            # Add geomean bar
            geomean_x = len(workloads)
            plt.bar(geomean_x - width, 1.0, width, color="lightgray", edgecolor="black")
            bar_proteus = plt.bar(geomean_x, np.exp(np.mean(np.log(improv_proteus))), width, color="#ff7f0e", edgecolor="black")
            bar_pim = plt.bar(geomean_x + width, geo_mean_improv_gpu, width, color="#1fc430", edgecolor="black")
            
            # Add height labels for geomean bars
            # for bar in bar_proteus:
                # height = bar.get_height()
                # plt.text(bar.get_x() + bar.get_width() / 2, height * 1.05, f"{height:.2f}", ha="center", va="bottom", fontsize=12)
            for bar in bar_pim:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2, height * 1.05, f"{height:.2f}", ha="center", va="bottom", fontsize=12)
            
            plt.xticks(list(x) + [geomean_x], workloads + ["Geomean"], rotation=30, ha="right", fontsize=12)
            geomean_x = len(workloads)
            plt.bar(geomean_x - width, 1.0, width, color="#cccccc", edgecolor="black")
            plt.bar(geomean_x, np.exp(np.mean(np.log(improv_proteus))), width, color="#ff7f0e", edgecolor="black")
            plt.bar(geomean_x + width, geo_mean_improv_gpu, width, color="#1fc430", edgecolor="black")
            plt.xticks(list(x) + [geomean_x], workloads + ["Geomean"], rotation=30, ha="right", fontsize=12)
    plt.axhline(y=1.0, color="black", linestyle="--", linewidth=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_file}")
    plt.show()

if __name__ == "__main__":
    main()
