import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_rows(csv_path: Path):
    df = pd.read_csv(csv_path)
    if len(df) < 3:
        raise ValueError(f"Expected at least 3 rows in {csv_path}, found {len(df)}")
    return df.columns.tolist(), df.iloc[0].values, df.iloc[1].values, df.iloc[2].values


def main():
    parser = argparse.ArgumentParser(
        description="Half-column plot for perf/area (throughput bars + TTFT lines)."
    )
    parser.add_argument(
        "--throughput",
        type=str,
        required=True,
        help="Perf/area CSV derived from decode throughput.",
    )
    parser.add_argument(
        "--prefill",
        type=str,
        required=True,
        help="Perf/area CSV derived from prefill TTFT latencies.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="perf_per_area_speedup_e2e.pdf",
        help="Output PDF filename.",
    )
    args = parser.parse_args()

    thr_workloads, h100_thr, proteus_thr, pim_thr = _load_rows(Path(args.throughput))
    pref_workloads, h100_pref, proteus_pref, pim_pref = _load_rows(Path(args.prefill))
    if thr_workloads != pref_workloads:
        raise ValueError("Throughput and prefill CSVs must share identical workloads/order.")
    workloads = thr_workloads

    # === Normalize to H100 baseline ===
    thr_speedup_gpu = np.ones_like(h100_thr, dtype=float)
    thr_speedup_proteus = np.divide(proteus_thr, h100_thr)
    thr_speedup_pim = np.divide(pim_thr, h100_thr)
    pref_speedup_proteus = np.divide(proteus_pref, h100_pref)
    pref_speedup_pim = np.divide(pim_pref, h100_pref)

    # === Plot ===
    plt.figure(figsize=(2.3, 0.8))
    ax = plt.gca()
    x = np.arange(len(workloads))
    width = 0.25

    bars_gpu = ax.bar(
        x - width,
        thr_speedup_gpu,
        width,
        label="H100",
        color="#b8b0b0",
        edgecolor=None,
    )
    bars_proteus = ax.bar(
        x,
        thr_speedup_proteus,
        width,
        label="Token TP (Proteus)",
        color="#55A868",
        edgecolor=None,
    )
    bars_pim = ax.bar(
        x + width,
        thr_speedup_pim,
        width,
        label="Token TP (DREAM)",
        color="#4C72B0",
        edgecolor=None,
    )

    ax.plot(
        x,
        pref_speedup_proteus,
        color="#89C79C",
        marker="o",
        markersize=3.5,
        linewidth=1.1,
        label="Prefill TP (Proteus)",
    )
    ax.plot(
        x + width,
        pref_speedup_pim,
        color="#82A7E0",
        marker="s",
        markersize=3.5,
        linewidth=1.1,
        label="Prefill TP (DREAM)",
    )

    ax.set_yscale("log")
    ax.set_ylabel("Perf./area vs H100", fontsize=6.5)
    ax.axhline(1, color="black", linestyle="--", linewidth=0.6)

    # === Geometric means ===
    def _geo_mean(arr):
        arr = np.asarray(arr, dtype=float)
        arr = arr[arr > 0]
        if len(arr) == 0:
            return 0.0
        return float(np.exp(np.mean(np.log(arr))))

    def _safe_min(arr):
        arr = np.asarray(arr, dtype=float)
        arr = arr[arr > 0]
        return arr.min() if len(arr) else 1.0

    geo_thr_gpu = _geo_mean(thr_speedup_gpu)
    geo_thr_proteus = _geo_mean(thr_speedup_proteus)
    geo_thr_pim = _geo_mean(thr_speedup_pim)
    geo_pref_proteus = _geo_mean(pref_speedup_proteus)
    geo_pref_pim = _geo_mean(pref_speedup_pim)

    x_geo = len(workloads)
    geo_bars_gpu = ax.bar(
        x_geo - width,
        geo_thr_gpu,
        width,
        color="#b8b0b0",
        edgecolor=None,
    )
    geo_bars_proteus = ax.bar(
        x_geo,
        geo_thr_proteus,
        width,
        color="#55A868",
        edgecolor=None,
    )
    geo_bars_pim = ax.bar(
        x_geo + width,
        geo_thr_pim,
        width,
        color="#4C72B0",
        edgecolor=None,
    )
    ax.plot(
        [x_geo],
        [geo_pref_proteus],
        marker="o",
        color="#89C79C",
        markersize=3.5,
    )
    ax.plot(
        [x_geo + width],
        [geo_pref_pim],
        marker="s",
        color="#82A7E0",
        markersize=3.5,
    )

    # === Labels for DREAM bars ===
    for bar in list(bars_pim) + list(geo_bars_pim):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height * 1.05,
            f"{height:.1f}",
            fontsize=4,
            ha="center",
            va="bottom",
        )

    for xi, val in zip(
        list(x + width) + [x_geo + width],
        list(pref_speedup_pim) + [geo_pref_pim],
    ):
        ax.text(
            xi,
            val * 3,
            f"{val:.1f}",
            fontsize=4,
            ha="center",
            va="bottom",
        )

    xticks = list(x) + [x_geo]
    ax.set_xticks(xticks, workloads + ["Geomean"], rotation=25, ha="right", fontsize=5.5)
    ax.tick_params(axis="y", labelsize=5)
    ax.legend(loc="upper left", fontsize=5, frameon=False, handlelength=1.4, ncol=2, bbox_to_anchor=(0.0, 1.55))
    ymax = max(
        np.max(thr_speedup_gpu),
        np.max(thr_speedup_pim),
        np.max(thr_speedup_proteus),
        np.max(pref_speedup_pim),
        np.max(pref_speedup_proteus),
    )
    ymin = min(
        _safe_min(thr_speedup_gpu),
        _safe_min(thr_speedup_pim),
        _safe_min(thr_speedup_proteus),
        _safe_min(pref_speedup_pim),
        _safe_min(pref_speedup_proteus),
    )
    ax.set_ylim(ymin / 10, max(ymax * 8, 10))

    plt.savefig(args.output, bbox_inches="tight")
    print(f"Saved perf/area e2e plot to {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
