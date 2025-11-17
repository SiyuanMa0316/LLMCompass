import csv
import argparse
def normalize_workload_name(name: str) -> str:
    """Normalize workload identifiers so that naming is consistent across sources."""
    normalized = name.replace("_transformer", "")
    normalized = normalized.replace("GEMM_1x", "GEMV_1x")
    normalized = normalized.replace("1024x12288x12288", "small")
    normalized = normalized.replace("2048x24576x24576", "large")
    normalized = normalized.replace("1x12288x12288", "small")
    normalized = normalized.replace("1x24576x24576", "large")
    return normalized


def display_name(name: str) -> str:
    return name.replace("Llama-3.1", "Llama3").replace("_", "-")

parser = argparse.ArgumentParser(description="Extract latencies from simulation output")
parser.add_argument("--input", type=str, default="run_all_output_10-3", help="Path to the input log file")

args = parser.parse_args()

input_file = args.input
output_file = f"latencies_{args.input}.csv"

workloads = []
latencies = []
h100_latencies = [
    0.0002655339506,
    0.0020424524691358,
    0.000045049255441008,
    0.000180182359679266,
    0.35947,
    238.99,
    0.02766,
    27.39910,
    0.13564,
    102.8096,
    0.02542602351,
    26.05659218,
]

with open(input_file, "r") as f:
    for line in f:
        if line.startswith("simulated latency:"):
            parts = line.strip().replace("simulated latency:", "").split()
            if len(parts) == 2:
                workload = normalize_workload_name(parts[0])
                latency = float(parts[1])
                workloads.append(workload)
                latencies.append(latency)

proteus_latencies_raw = {
    "GEMM_1024x12288x12288": 95567.21 / 1000,
    "GEMM_2048x24576x24576": 382268.84 / 1000,
    "GEMV_1x12288x12288": 88.74 / 1000,
    "GEMV_1x24576x24576": 177.47 / 1000,
    "gpt3-175B_prefill": 1401086.566 * 96 / 1000,
    "gpt3-175B_decode": 1306.107936 * 96 * 2048 / 1000,
    "gpt3-6.7B_prefill": 572829.6539 * 32 / 1000,
    "gpt3-6.7B_decode": 532.46928 * 32 * 2048 / 1000,
    "Llama-3.1-70B_prefill": 1113802.157 * 80 / 1000,
    "Llama-3.1-70B_decode": 1035.359808 * 80 * 2048 / 1000,
    "Llama-3.1-8B_prefill": 556901.138 * 32 / 1000,
    "Llama-3.1-8B_decode": 517.679904 * 32 * 2048 / 1000,
}

proteus_latencies = {
    normalize_workload_name(name): value for name, value in proteus_latencies_raw.items()
}

desired_order = [
    "gpt3-175B_prefill",
    "Llama-3.1-70B_prefill",
    "Llama-3.1-8B_prefill",
    "gpt3-6.7B_prefill",
    "gpt3-175B_decode",
    "Llama-3.1-70B_decode",
    "Llama-3.1-8B_decode",
    "gpt3-6.7B_decode",
]

# Build lookup tables from the original ordering
sim_latency_map = dict(zip(workloads, latencies))
h100_latency_map = dict(zip(workloads, h100_latencies))

for name in desired_order:
    if name not in sim_latency_map:
        raise ValueError(f"Workload '{name}' not found in simulation latency log")
    if name not in h100_latency_map:
        raise ValueError(f"Workload '{name}' missing in H100 latency list")
    if name not in proteus_latencies:
        raise ValueError(f"Workload '{name}' missing in Proteus latency table")

workloads = desired_order
h100_latencies = [h100_latency_map[name] for name in desired_order]
proteus_latencies_ordered = [proteus_latencies[name] for name in desired_order]
latencies = [sim_latency_map[name] for name in desired_order]

display_workloads = [display_name(name) for name in workloads]
print(display_workloads)

# write CSV: workload in header, latency rows follow
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(display_workloads)
    writer.writerow(h100_latencies)
    writer.writerow(proteus_latencies_ordered)
    writer.writerow(latencies)

print(f"Saved {len(workloads)} workloads to {output_file}")