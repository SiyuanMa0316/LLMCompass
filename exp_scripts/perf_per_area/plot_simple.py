import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from matplotlib.patches import Patch

FIGSIZE = (10, 6)
XTICK_FONTSIZE = 13
YLABEL_FONTSIZE = 18
LEGEND_FONTSIZE = 14
ANNOT_FONTSIZE = 12
AXHLINE_WIDTH = 1.0

def geom_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if np.any(values <= 0):
        raise ValueError("Geometric mean requires strictly positive values.")
    return float(np.exp(np.mean(np.log(values))))

def main():
    parser = argparse.ArgumentParser(
        description="Plot relative perf/area improvement (Proteus, PIM vs H100)"
    )
    parser.add_argument("--input", required=False, help="Path to input CSV file")
    args = parser.parse_args()

    input_file = args.input
    output_file = os.path.splitext(input_file)[0] + "_speedup_column.pdf"

    df = pd.read_csv(input_file)
    h100 = df.iloc[0].values
    pim = df.iloc[2].values
    proteus = df.iloc[1].values
    workloads = df.columns.tolist()

    improv_h100 = np.ones_like(h100, dtype=float)
    improv_proteus = proteus / h100
    improv_pim = pim / h100
    improv_pim_vs_proteus = pim / proteus

    prefill_slice = slice(0, len(workloads) // 2)
    decode_slice = slice(len(workloads) // 2, len(workloads))

    geo_prefill = {
        "H100": 1.0,
        "Proteus": geom_mean(improv_proteus[prefill_slice]),
        "PIM": geom_mean(improv_pim[prefill_slice]),
    }
    geo_decode = {
        "H100": 1.0,
        "Proteus": geom_mean(improv_proteus[decode_slice]),
        "PIM": geom_mean(improv_pim[decode_slice]),
    }
    geo_overall = {
        "H100": 1.0,
        "Proteus": geom_mean(improv_proteus),
        "PIM": geom_mean(improv_pim),
    }

    print(f"Geomean (Prefill)  PIM vs Proteus: {geom_mean(improv_pim_vs_proteus[prefill_slice]):.2f}×")
    print(f"Geomean (Decode)   PIM vs Proteus: {geom_mean(improv_pim_vs_proteus[decode_slice]):.2f}×")
    print(f"Geomean (Overall)  PIM vs Proteus: {geom_mean(improv_pim_vs_proteus):.2f}×")
    print(f"Geomean (Prefill)  PIM vs H100:    {geo_prefill['PIM']:.2f}×")
    print(f"Geomean (Decode)   PIM vs H100:    {geo_decode['PIM']:.2f}×")
    print(f"Geomean (Overall)  PIM vs H100:    {geo_overall['PIM']:.2f}×")

    extra_labels = ["Prefill_Geomean", "Decode_Geomean", "Overall_Geomean"]

    workloads_all: list[str] = []
    values_h100: list[float] = []
    values_proteus: list[float] = []
    values_pim: list[float] = []

    def extend_segment(start: int, end: int) -> None:
        workloads_all.extend(workloads[start:end])
        values_h100.extend(improv_h100[start:end])
        values_proteus.extend(improv_proteus[start:end])
        values_pim.extend(improv_pim[start:end])

    extend_segment(0, prefill_slice.stop)
    workloads_all.append(extra_labels[0])
    values_h100.append(geo_prefill["H100"])
    values_proteus.append(geo_prefill["Proteus"])
    values_pim.append(geo_prefill["PIM"])

    extend_segment(prefill_slice.stop, decode_slice.stop)
    workloads_all.append(extra_labels[1])
    values_h100.append(geo_decode["H100"])
    values_proteus.append(geo_decode["Proteus"])
    values_pim.append(geo_decode["PIM"])

    workloads_all.append(extra_labels[2])
    values_h100.append(geo_overall["H100"])
    values_proteus.append(geo_overall["Proteus"])
    values_pim.append(geo_overall["PIM"])

    values_h100 = np.asarray(values_h100, dtype=float)
    values_proteus = np.asarray(values_proteus, dtype=float)
    values_pim = np.asarray(values_pim, dtype=float)

    plt.figure(figsize=FIGSIZE)
    x = np.arange(len(workloads_all))

    geomean_indices = [i for i, label in enumerate(workloads_all) if "Geomean" in label]
    geomean_mask = np.array([idx in geomean_indices for idx in range(len(workloads_all))], dtype=bool)
    non_geomean_mask = ~geomean_mask

    colors_h100 = np.full(len(workloads_all), "#838181", dtype=object)
    colors_proteus = np.full(len(workloads_all), "#ee974b", dtype=object)
    colors_pim = np.full(len(workloads_all), "#32a73e", dtype=object)

    width = 0.25
    bars_h100 = plt.bar(
        x[non_geomean_mask] - width,
        values_h100[non_geomean_mask],
        width,
        label="H100",
        color=colors_h100[non_geomean_mask],
        edgecolor="black",
    )
    bars_proteus = plt.bar(x, values_proteus, width, label="Proteus", color=colors_proteus, edgecolor="black")
    bars_pim = plt.bar(x + width, values_pim, width, label="DREAM(Ours)", color=colors_pim, edgecolor="black")

    for idx, bar in enumerate(bars_proteus):
        if geomean_mask[idx]:
            bar.set_hatch("/")
            bar.set_edgecolor((0, 0, 0, 0.5))

    for idx, bar in enumerate(bars_pim):
        if geomean_mask[idx]:
            bar.set_hatch("/")
            bar.set_edgecolor((0, 0, 0, 0.5))

    plt.xticks(x, workloads_all, rotation=15, ha="right", fontsize=XTICK_FONTSIZE, color="black")
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(axis="both", labelsize=XTICK_FONTSIZE)
    for tick in ax.get_xticklabels():
        tick.set_color("black")

    plt.yscale("log")
    plt.ylabel("Normalized Perf./ Area", fontsize=YLABEL_FONTSIZE)
    plt.axhline(y=1.0, color="black", linestyle="--", linewidth=AXHLINE_WIDTH)

    # for idx, tick in enumerate(plt.gca().get_xticklabels()):
    #     if idx in geomean_indices:
    #         tick.set_color("#43A857")

    for bar, idx in zip(bars_pim, range(len(workloads_all))):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height * 1.05,
            f"x{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=ANNOT_FONTSIZE,
            # color="#314a3e" if idx in geomean_indices else "black",
        )

    handles, labels = plt.gca().get_legend_handles_labels()
    if geomean_indices:
        proteus_geo_patch = Patch(
            facecolor=colors_proteus[geomean_indices[0]],
            edgecolor=(0, 0, 0, 0.45),
            hatch="/",
            label="Proteus Geomean",
        )
        dream_geo_patch = Patch(
            facecolor=colors_pim[geomean_indices[0]],
            edgecolor=(0, 0, 0, 0.45),
            hatch="/",
            label="DREAM Geomean",
        )
        handles = list(handles) + [proteus_geo_patch, dream_geo_patch]
        labels = list(labels) + ["Proteus Geomean", "DREAM Geomean"]
    plt.legend(handles, labels, fontsize=LEGEND_FONTSIZE, frameon=False)

    plt.tight_layout()
    plt.savefig(f"{output_file}")
    plt.show()

if __name__ == "__main__":
    main()
