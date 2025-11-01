#!/usr/bin/env python3
import re
import csv
import argparse
from functools import reduce
from operator import mul

parser = argparse.ArgumentParser(description="Extract latencies & PE utilization from simulation output")
parser.add_argument("--input", type=str, default="run_gemv_output", help="Path to the input log file")
args = parser.parse_args()

input_file = args.input
output_file = f"latencies_{args.input}.csv"

# ---------- Helpers ----------
ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")  # strip ANSI escape codes
NUM_RE  = r"[-+]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"

SIM_LINE = re.compile(rf"^simulated latency:\s*(?P<workload>\S+)\s+(?P<sim>{NUM_RE})\s*$")
TOTAL_RE = re.compile(rf"^\|\s*Total Latency\s*\|\s*(?P<total>{NUM_RE})")
COMP_RE  = re.compile(rf"^\|\s*Total Compute Latency\s*\|\s*(?P<compute>{NUM_RE})")
IO_RE    = re.compile(rf"^\|\s*IO Latency\s*\|\s*(?P<io>{NUM_RE})")

def to_float(x):
    try:
        return float(x)
    except Exception:
        return None

def is_sep_row(line: str) -> bool:
    # table separator like: |-------------|----| ... |
    s = line.strip()
    return s.startswith("|") and set(s.replace("|", "").replace("+", "").strip()) <= {"-", " "}

def parse_table_row(line: str) -> list[str]:
    # split a markdown-style table row -> list of stripped cells (excluding edges)
    parts = [c.strip() for c in line.strip().split("|")]
    if len(parts) >= 3:
        return parts[1:-1]  # drop leading/trailing empties from edges
    return []

def product(vals):
    return reduce(mul, vals, 1.0)

def compute_pe(util, lats):
    keys = ["Col", "Array", "Bank", "Device", "Rank", "Channel"]
    if lats.get("total") in (None, 0) or lats.get("compute") is None:
        return None
    if any(util.get(k) is None for k in keys):
        return None
    return product([util[k] for k in keys]) * (lats["compute"] / lats["total"])

# ---------- Scan state ----------
current_utils = {k: None for k in ["Row","Col","Array","Bank","Device","Rank","Channel"]}
current_lats  = {"total": None, "compute": None, "io": None}
records = []

expect_util_header = False
expect_util_data = False
util_header_cols: list[str] = []

with open(input_file, "r", encoding="utf-8", errors="ignore") as fh:
    for raw in fh:
        line = ANSI_RE.sub("", raw.rstrip("\n"))

        # 1) utilization table: detect header row, then grab next data row
        if " Utilization " in line or line.strip().startswith("--------------- Utilization"):
            expect_util_header = True
            expect_util_data = False
            util_header_cols = []
            continue

        if expect_util_header and line.strip().startswith("|") and not is_sep_row(line):
            cols = parse_table_row(line)
            # We need exactly these columns (order can vary—map by name)
            target = {"Row","Col","Array","Bank","Device","Rank","Channel"}
            if set(cols) >= target:   # header contains our targets
                util_header_cols = cols
                expect_util_header = False
                # Next non-separator row should be the numeric row
                expect_util_data = True
            else:
                # Not the header we want; keep waiting
                pass
            continue

        if expect_util_data:
            if is_sep_row(line) or not line.strip().startswith("|"):
                # still skipping separators / not yet data
                continue
            # numeric row
            data_cells = parse_table_row(line)
            # build a map header->cell
            hdr_to_val = {h: data_cells[i] for i, h in enumerate(util_header_cols) if i < len(data_cells)}
            # pull required ones
            for k in ["Row","Col","Array","Bank","Device","Rank","Channel"]:
                v = to_float(hdr_to_val.get(k))
                current_utils[k] = v
            expect_util_data = False
            # do not 'continue' so the same line can also match latency patterns if present

        # 2) latencies block
        m = TOTAL_RE.match(line)
        if m:
            current_lats["total"] = to_float(m.group("total"))
        m = COMP_RE.match(line)
        if m:
            current_lats["compute"] = to_float(m.group("compute"))
        m = IO_RE.match(line)
        if m:
            current_lats["io"] = to_float(m.group("io"))

        # 3) simulated latency => snapshot a record
        m = SIM_LINE.match(line.strip())
        if m:
            workload = m.group("workload")
            sim_lat = to_float(m.group("sim"))
            pe = compute_pe(current_utils, current_lats)
            rec = {
                "workload": workload,
                "simulated_latency": sim_lat,
                "total_latency": current_lats["total"],
                "compute_latency": current_lats["compute"],
                "io_latency": current_lats["io"],
                "util_row": current_utils["Row"],
                "util_col": current_utils["Col"],
                "util_array": current_utils["Array"],
                "util_bank": current_utils["Bank"],
                "util_device": current_utils["Device"],
                "util_rank": current_utils["Rank"],
                "util_channel": current_utils["Channel"],
                "pe_utilization": pe,
            }
            records.append(rec)
            # reset if each workload has its own block; comment out if values should carry forward
            current_utils = {k: None for k in current_utils}
            current_lats  = {"total": None, "compute": None, "io": None}
            expect_util_header = False
            expect_util_data = False
            util_header_cols = []

# ---------- Write CSV ----------
headers = [
    "workload",
    "simulated_latency",
    "total_latency",
    "compute_latency",
    "io_latency",
    "util_col",
    "util_array",
    "util_bank",
    "util_device",
    "util_rank",
    "util_channel",
    "util_row",
    "pe_utilization",
]

with open(output_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    for r in records:
        writer.writerow({k: ("" if r.get(k) is None else r[k]) for k in headers})

print(f"Saved {len(records)} workloads to {output_file}")
