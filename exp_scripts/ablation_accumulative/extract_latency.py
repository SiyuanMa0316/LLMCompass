import csv
import argparse

def normalize_workload_name(name: str) -> str:
    normalized = name.replace("_transformer", "")
    normalized = normalized.replace("GEMM_1x", "GEMV_1x")
    normalized = normalized.replace("1024x12288x12288", "small")
    normalized = normalized.replace("2048x24576x24576", "large")
    normalized = normalized.replace("1x12288x12288", "small")
    normalized = normalized.replace("1x24576x24576", "large")
    return normalized


def display_name(name: str) -> str:
    return name.replace("Llama-3.1", "Llama3").replace("_", "-")


parser = argparse.ArgumentParser(description="Extract latencies from ablation runs")
args = parser.parse_args()

input_files = [
    "run_all_output_base",
    "run_all_output_no_popcount",
    "run_all_output_no_popcount_no_broadcast",
    "run_all_output_no_popcount_no_broadcast_no_locality_buffer",
]
output_file = "latencies_ablation.csv"

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

H100_LATENCIES = {
    "gpt3-175B_prefill": 0.0002655339506,
    "Llama-3.1-70B_prefill": 0.0020424524691358,
    "Llama-3.1-8B_prefill": 0.000045049255441008,
    "gpt3-6.7B_prefill": 0.000180182359679266,
    "gpt3-175B_decode": 0.35947,
    "Llama-3.1-70B_decode": 238.99,
    "Llama-3.1-8B_decode": 0.02766,
    "gpt3-6.7B_decode": 27.39910,
}

runs: list[dict[str, float]] = []

for input_path in input_files:
    latency_map: dict[str, float] = {}
    with open(input_path, "r") as f:
        for line in f:
            if line.startswith("simulated latency:"):
                parts = line.strip().replace("simulated latency:", "").split()
                if len(parts) == 2:
                    workload = normalize_workload_name(parts[0])
                    latency = float(parts[1])
                    latency_map[workload] = latency
    runs.append(latency_map)

for workload in desired_order:
    if workload not in H100_LATENCIES:
        raise ValueError(f"H100 latency missing for workload '{workload}'")
    for input_path, run in zip(input_files, runs):
        if workload not in run:
            raise ValueError(f"Workload '{workload}' not found in run '{input_path}'")

display_workloads = [display_name(name) for name in desired_order]
h100_row = [H100_LATENCIES[name] for name in desired_order]
ablation_rows = [[run[name] for name in desired_order] for run in runs]

with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(display_workloads)
    writer.writerow(h100_row)
    for row in ablation_rows:
        writer.writerow(row)

print(f"Saved {len(desired_order)} workloads to {output_file}")