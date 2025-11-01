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
    parser.add_argument("--output", default=None, help="Output PDF filename (optional)")
    parser.add_argument("--annotate", action="store_true", help="Add numeric labels above bars")
    args = parser.parse_args()

    input_file = args.input
    output_file = args.output or os.path.splitext(input_file)[0] + "_speedup_column.pdf"

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
    pim_delta = improv_pim - 1
    proteus_delta = improv_proteus - 1

    # === Plot grouped bars ===
    x = np.arange(len(workloads))
    width = 0.35
    plt.figure(figsize=(3.4, 2.8))  # single-column width, slightly taller

    bars_proteus = plt.bar(
        x - width / 2, proteus_delta, width, bottom=1, label="Proteus / H100", color="#55A868"
    )
    bars_pim = plt.bar(
        x + width / 2, pim_delta, width, bottom=1, label="PIM / H100", color="#4C72B0"
    )

    # === Style ===
    plt.yscale("log")
    plt.ylabel("Normalized Perf/Area", fontsize=7)
    plt.xticks(x, workloads, rotation=45, ha="right", fontsize=6)
    plt.legend(fontsize=6, frameon=False)
    plt.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.4)
    plt.title("Perf/Area Improvement", fontsize=10, pad=3)

    # --- Baseline at 1× ---
    plt.axhline(y=1, color="gray", linestyle="--", linewidth=0.6)
    plt.text(-0.4, 1.05, "H100 baseline", color="gray", fontsize=5, va="bottom")

    # --- Helper for formatting values ---
    def fmt_val(v):
        val = v + 1
        return f"{val:.1e}" if val < 0.1 or val > 1000 else f"{val:.2f}"

    # --- Annotate bars if requested ---
    if args.annotate:
        def annotate_bars(bars, deltas):
            for bar, delta in zip(bars, deltas):
                val = delta + 1
                if val >= 1:
                    y = val * 1.1
                    va = "bottom"
                else:
                    y = val * 0.9
                    va = "top"
                y = max(y, 1e-3)
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    y,
                    fmt_val(delta),
                    ha="center",
                    va=va,
                    fontsize=5,
                    clip_on=False,
                )
        annotate_bars(bars_proteus, proteus_delta)
        annotate_bars(bars_pim, pim_delta)

    # === Adjust limits ===
    ymin = min(min(improv_pim.min(), improv_proteus.min()) * 0.8, 0.1)
    ymax = max(max(improv_pim.max(), improv_proteus.max()) * 1.2, 10)
    plt.ylim(ymin, ymax)

    plt.tight_layout(pad=0.7)
    plt.savefig(output_file, bbox_inches="tight", dpi=600)
    print(f"Saved plot to {output_file}")

if __name__ == "__main__":
    main()
