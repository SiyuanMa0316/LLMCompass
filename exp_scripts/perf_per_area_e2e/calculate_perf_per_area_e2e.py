import argparse
import csv
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_COMP_DIR = (SCRIPT_DIR / ".." / "comparison_e2e").resolve()

H100_AREA_MM2 = 5740  # scaled H100 die + flattened HBM area
PIM_PERIPHERAL_AREA_MM2 = 830.6 
# PIM_PERIPHERAL_AREA_MM2 = 5740
PROTEUS_DRAM_AREA_MM2 = 8 * 8 / 16 * 66
PROTEUS_PERIPHERAL_AREA_MM2 = PROTEUS_DRAM_AREA_MM2 * 0.01
OPS_PER_BATCH = 1e6  # constant factor, cancels out across comparisons


def _resolve(path: Optional[str]) -> Optional[Path]:
    return None if path is None else Path(path).expanduser().resolve()


def _load_three_rows(csv_path: Path):
    df = pd.read_csv(csv_path)
    if len(df) < 3:
        raise ValueError(f"Expected at least 3 rows in {csv_path}, found {len(df)}")
    return df.columns.tolist(), df.iloc[0].values, df.iloc[1].values, df.iloc[2].values


def _latency_to_perf_per_area(latencies, area_mm2):
    latencies = np.asarray(latencies, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        perf = np.where(latencies > 0, OPS_PER_BATCH / latencies / area_mm2, 0.0)
    return perf.tolist()


def _throughput_to_perf_per_area(throughputs, area_mm2):
    throughputs = np.asarray(throughputs, dtype=float)
    return np.where(throughputs > 0, throughputs / area_mm2, 0.0).tolist()


def _write_csv(out_path: Path, workloads, h100_row, proteus_row, pim_row):
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(workloads)
        writer.writerow(h100_row)
        writer.writerow(proteus_row)
        writer.writerow(pim_row)
    print(f"Wrote perf/area values to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute perf/area datasets for e2e workloads (throughput + TTFT)."
    )
    parser.add_argument(
        "--latencies",
        type=str,
        default=str(DEFAULT_COMP_DIR / "latencies_e2e_1024to128_run_all_output_11-8.csv"),
        help="CSV with end-to-end latencies (seconds).",
    )
    parser.add_argument(
        "--throughput",
        type=str,
        default=str(DEFAULT_COMP_DIR / "throughput_decode_1024to128_run_all_output_11-8.csv"),
        help="CSV with decode throughput (tokens/s).",
    )
    parser.add_argument(
        "--prefill",
        type=str,
        default=str(DEFAULT_COMP_DIR / "latencies_prefill_1024to128_run_all_output_11-8.csv"),
        help="CSV with prefill-only latencies (seconds).",
    )
    parser.add_argument("--latency-output", type=str, help="Override output path for e2e latency perf/area CSV.")
    parser.add_argument("--throughput-output", type=str, help="Override output path for throughput perf/area CSV.")
    parser.add_argument("--prefill-output", type=str, help="Override output path for prefill perf/area CSV.")
    args = parser.parse_args()

    lat_path = _resolve(args.latencies)
    thr_path = _resolve(args.throughput)
    pref_path = _resolve(args.prefill)
    if lat_path is None or not lat_path.exists():
        raise FileNotFoundError(f"Missing latencies CSV: {lat_path}")
    if thr_path is None or not thr_path.exists():
        raise FileNotFoundError(f"Missing throughput CSV: {thr_path}")
    if pref_path is None or not pref_path.exists():
        raise FileNotFoundError(f"Missing prefill CSV: {pref_path}")

    # === E2E latency perf/area ===
    workloads, h100_lat, proteus_lat, pim_lat = _load_three_rows(lat_path)
    h100_perf_lat = _latency_to_perf_per_area(h100_lat, H100_AREA_MM2)
    proteus_perf_lat = _latency_to_perf_per_area(proteus_lat, PROTEUS_PERIPHERAL_AREA_MM2)
    pim_perf_lat = _latency_to_perf_per_area(pim_lat, PIM_PERIPHERAL_AREA_MM2)

    lat_output = _resolve(args.latency_output) or (SCRIPT_DIR / f"perf_per_area_{lat_path.stem}.csv")
    _write_csv(lat_output, workloads, h100_perf_lat, proteus_perf_lat, pim_perf_lat)

    # === Throughput perf/area ===
    workloads_thr, h100_thr, proteus_thr, pim_thr = _load_three_rows(thr_path)
    h100_perf_thr = _throughput_to_perf_per_area(h100_thr, H100_AREA_MM2)
    proteus_perf_thr = _throughput_to_perf_per_area(proteus_thr, PROTEUS_PERIPHERAL_AREA_MM2)
    pim_perf_thr = _throughput_to_perf_per_area(pim_thr, PIM_PERIPHERAL_AREA_MM2)

    thr_output = _resolve(args.throughput_output) or (SCRIPT_DIR / f"perf_per_area_{thr_path.stem}.csv")
    _write_csv(thr_output, workloads_thr, h100_perf_thr, proteus_perf_thr, pim_perf_thr)

    # === Prefill perf/area ===
    workloads_pref, h100_pref, proteus_pref, pim_pref = _load_three_rows(pref_path)
    h100_perf_pref = _latency_to_perf_per_area(h100_pref, H100_AREA_MM2)
    proteus_perf_pref = _latency_to_perf_per_area(proteus_pref, PROTEUS_PERIPHERAL_AREA_MM2)
    pim_perf_pref = _latency_to_perf_per_area(pim_pref, PIM_PERIPHERAL_AREA_MM2)

    pref_output = _resolve(args.prefill_output) or (SCRIPT_DIR / f"perf_per_area_{pref_path.stem}.csv")
    _write_csv(pref_output, workloads_pref, h100_perf_pref, proteus_perf_pref, pim_perf_pref)


if __name__ == "__main__":
    main()
