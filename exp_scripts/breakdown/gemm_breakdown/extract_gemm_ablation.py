#!/usr/bin/env python3
"""
Extract GEMM latency breakdowns from log files into a CSV suitable for ablation plots.
"""
import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List

METRIC_PATTERNS = {
    "total_latency": re.compile(r"Total Latency\s*\|\s*([0-9.eE+-]+)"),
    "array_latency": re.compile(r"Total Array Latency\s*\|\s*([0-9.eE+-]+)"),
    "reduction_latency": re.compile(r"Total Reduction Latency\s*\|\s*([0-9.eE+-]+)"),
    "io_latency": re.compile(r"IO Latency\s*\|\s*([0-9.eE+-]+)"),
}


def parse_log(path: Path) -> Dict[str, float]:
    text = path.read_text()
    metrics: Dict[str, float] = {}
    for key, pattern in METRIC_PATTERNS.items():
        match = pattern.search(text)
        if not match:
            raise ValueError(f"Failed to find '{key}' in {path}")
        metrics[key] = float(match.group(1))
    return metrics


def parse_inputs(args_inputs: List[str]) -> List[Dict[str, Path]]:
    parsed = []
    for item in args_inputs:
        if "=" in item:
            label, path_str = item.split("=", 1)
        else:
            path_str = item
            label = Path(path_str).name
        path = Path(path_str).expanduser().resolve()
        parsed.append({"label": label, "path": path})
    return parsed


def main():
    parser = argparse.ArgumentParser(
        description="Extract GEMM latency breakdowns into gemm_ablation.csv"
    )
    parser.add_argument(
        "--inputs",
        "-i",
        nargs="+",
        required=True,
        help="Input log files, optionally specified as label=path (e.g., base=run_gemm_output).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="gemm_ablation.csv",
        help="Output CSV path (default: gemm_ablation.csv).",
    )
    args = parser.parse_args()

    entries = parse_inputs(args.inputs)
    rows = []
    for entry in entries:
        metrics = parse_log(entry["path"])
        rows.append({"ablation": entry["label"], **metrics})

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["ablation", "total_latency", "array_latency", "reduction_latency", "io_latency"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
