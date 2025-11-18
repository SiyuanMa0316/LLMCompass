import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

BASE_FONT = 7
plt.rcParams.update({
    "font.size": BASE_FONT,
    "axes.labelsize": BASE_FONT + 1,
    "axes.titlesize": BASE_FONT,
    "legend.fontsize": BASE_FONT,
    "xtick.labelsize": BASE_FONT,
    "ytick.labelsize": BASE_FONT,
    "figure.dpi": 300,
})

FIG_WIDTH = 4
FIGSIZE = (FIG_WIDTH, 2.5 * 0.8)
ANNOT_FONTSIZE = BASE_FONT - 1
AXHLINE_WIDTH = 0.3
BAR_LINEWIDTH = 0.3
BAR_WIDTH = 0.25
SPINE_LINEWIDTH = 0.8

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

    geo_overall = {
        "H100": 1.0,
        "Proteus": geom_mean(improv_proteus),
        "PIM": geom_mean(improv_pim),
    }

    print(f"Geomean (Overall)  PIM vs Proteus: {geom_mean(improv_pim_vs_proteus):.2f}×")
    print(f"Geomean (Overall)  PIM vs H100:    {geo_overall['PIM']:.2f}×")

    extra_label = "Geomean"

    workloads_all: list[str] = []
    values_h100: list[float] = []
    values_proteus: list[float] = []
    values_pim: list[float] = []

    def extend_segment(start: int, end: int) -> None:
        workloads_all.extend(workloads[start:end])
        values_h100.extend(improv_h100[start:end])
        values_proteus.extend(improv_proteus[start:end])
        values_pim.extend(improv_pim[start:end])

    extend_segment(0, len(workloads))
    workloads_all.append(extra_label)
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

    bars_h100 = plt.bar(
        x[non_geomean_mask] - BAR_WIDTH,
        values_h100[non_geomean_mask],
        BAR_WIDTH,
        label="H100",
        color=colors_h100[non_geomean_mask],
        edgecolor="black",
        linewidth=BAR_LINEWIDTH,
    )
    bars_proteus = plt.bar(
        x,
        values_proteus,
        BAR_WIDTH,
        label="Proteus",
        color=colors_proteus,
        edgecolor="black",
        linewidth=BAR_LINEWIDTH,
    )
    bars_pim = plt.bar(
        x + BAR_WIDTH,
        values_pim,
        BAR_WIDTH,
        label="DREAM(Ours)",
        color=colors_pim,
        edgecolor="black",
        linewidth=BAR_LINEWIDTH,
    )

    plt.xticks(x, workloads_all, rotation=30, ha="right", fontsize=plt.rcParams["xtick.labelsize"], color="black")
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_LINEWIDTH)
    ax.tick_params(axis="x", labelsize=plt.rcParams["xtick.labelsize"])
    ax.tick_params(axis="y", labelsize=plt.rcParams["ytick.labelsize"])
    for tick in ax.get_xticklabels():
        tick.set_color("black")

    plt.yscale("log")
    ymin = (ax.get_ylim()[0])
    ymax = (ax.get_ylim()[1])*10
    ax.set_ylim(ymin, ymax)
    plt.ylabel("Normalized Perf./ Area")
    plt.axhline(y=1.0, color="black", linestyle="--", linewidth=AXHLINE_WIDTH)

    # for idx, tick in enumerate(plt.gca().get_xticklabels()):
    #     if idx in geomean_indices:
    #         tick.set_color("#43A857")

    for bar, idx in zip(bars_pim, range(len(workloads_all))):
        height = bar.get_height()
        plt.text(
            # bar.get_x() - bar.get_width() / 2,
            bar.get_x(),
            # height * 1.05 if idx %2 == 1 else  height * 1.2,
            height * 1.05,
            f"x{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=ANNOT_FONTSIZE,
            # color="#314a3e" if idx in geomean_indices else "black",
        )


    for bar, idx in zip(bars_proteus, range(len(workloads_all))):
        height = bar.get_height()
        if height > 1:
            plt.text(
                bar.get_x() - bar.get_width() / 3,
                # bar.get_x(),
                height * 1.05,
                f"x{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=ANNOT_FONTSIZE,
                # color="#314a3e" if idx in geomean_indices else "black",
            )

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=3,
    )

    plt.tight_layout(pad=0.35)
    plt.savefig(f"{output_file}", bbox_inches="tight", pad_inches=0.02)
    plt.show()

if __name__ == "__main__":
    main()
