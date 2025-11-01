#!/usr/bin/env python3
import re
import sys
import csv
import argparse
from pathlib import Path

PATTERN = re.compile(
    r"^\s*Tile mapping:\s*(\S+),\s*arr_mapping:\s*(\S+),\s*latency:\s*([0-9.+\-eE]+)"
)

def parse_file(fp):
    for line in fp:
        m = PATTERN.search(line)
        if m:
            tile, arr, lat = m.groups()
            # Try to parse latency as float; if it fails, keep raw text
            try:
                lat_val = float(lat)
            except ValueError:
                lat_val = lat
            yield (tile, arr, lat_val)

def main():
    ap = argparse.ArgumentParser(
        description="Extract Tile/Array mappings and latencies to CSV."
    )
    ap.add_argument("input", nargs="?", default="gemm_1024_12288_12288_mapping",
                    help="Input file (default: stdin)")
    ap.add_argument("-o", "--output", default="mapping_sensitivity.csv",
                    help="Output CSV file (default: stdout)")
    ap.add_argument("--no-header", action="store_true",
                    help="Do not write CSV header")
    ap.add_argument("--sort", choices=["lat", "tile", "arr"], default=None,
                    help="Sort rows by this column (ascending)")
    args = ap.parse_args()

    # Open input
    if args.input == "-":
        infile = sys.stdin
        close_in = False
    else:
        infile = open(args.input, "r", encoding="utf-8", errors="ignore")
        close_in = True

    rows = list(parse_file(infile))
    if close_in:
        infile.close()

    # Optional sort
    if args.sort:
        key_idx = {"tile": 0, "arr": 1, "lat": 2}[args.sort]
        rows.sort(key=lambda r: (r[key_idx] if key_idx != 2 else (float(r[2]) if isinstance(r[2], (int, float)) else float(str(r[2])))))

    # Open output
    if args.output == "-":
        out = sys.stdout
        close_out = False
    else:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        out = open(args.output, "w", newline="", encoding="utf-8")
        close_out = True

    writer = csv.writer(out)
    if not args.no_header:
        writer.writerow(["tile_mapping", "arr_mapping", "latency"])
    for row in rows:
        writer.writerow(row)

    if close_out:
        out.close()

if __name__ == "__main__":
    main()
