#!/usr/bin/env python3
import argparse
import csv
import re
import sys
from pathlib import Path

LOG_LINE = re.compile(
    r"^(?P<kernel>[A-Za-z0-9_]+)\s+latency:\s*(?P<latency>[0-9.eE+-]+),"
    r".*?array latency:\s*(?P<array>[0-9.eE+-]+),\s*"
    r"reduction latency:\s*(?P<reduction>[0-9.eE+-]+),\s*"
    r"io overhead:\s*(?P<io>[0-9.eE+-]+)"
)

def parse_log(text: str):
    for line in text.splitlines():
        match = LOG_LINE.search(line.strip())
        if match:
            yield {
                "kernel": match.group("kernel"),
                "latency": match.group("latency"),
                "array_latency": match.group("array"),
                "reduction_latency": match.group("reduction"),
                "io_overhead": match.group("io"),
            }

def main():
    parser = argparse.ArgumentParser(description="Extract kernel latency breakdowns from a log file.")
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to the transformer simulation log file.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    log_text = input_path.read_text()
    rows = list(parse_log(log_text))
    if not rows:
        print("No kernel lines found.", file=sys.stderr)
        sys.exit(2)

    base_name = input_path.name
    if base_name.lower().endswith(".csv"):
        base_name = base_name[:-4]
    else:
        base_name = input_path.stem
    out_path = input_path.parent / f"kernel_latencies_{base_name}.csv"

    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["kernel", "latency", "array_latency", "reduction_latency", "io_overhead"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")

if __name__ == "__main__":
    main()
