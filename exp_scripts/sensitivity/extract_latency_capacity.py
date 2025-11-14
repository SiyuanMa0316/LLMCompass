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


parser = argparse.ArgumentParser(description="Extract latencies from simulation output")
args = parser.parse_args()

input_files = [
    "run_all_output_base",
    "run_all_output_int4",
    "run_all_output_256g",
    "run_all_output_64g",
    "run_all_output_8g",
]
output_file = "latencies_sensitivity_capacity.csv"

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

runs = []
workload_order_from_base: list[str] = []

for file_idx, input_path in enumerate(input_files):
    latency_map: dict[str, float] = {}
    with open(input_path, "r") as f:
        for line in f:
            if line.startswith("simulated latency:"):
                parts = line.strip().replace("simulated latency:", "").split()
                if len(parts) == 2:
                    workload = normalize_workload_name(parts[0])
                    latency_map[workload] = float(parts[1])
                    if file_idx == 0:
                        workload_order_from_base.append(workload)
    runs.append(latency_map)

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

if len(workload_order_from_base) != len(h100_latencies):
    raise ValueError("Mismatch between base-run workloads and H100 latency list length.")

h100_map = {
    workload_order_from_base[idx]: h100_latencies[idx] for idx in range(len(workload_order_from_base))
}

for name in desired_order:
    if name not in h100_map:
        raise ValueError(f"H100 latency missing for workload '{name}'")
    for run in runs:
        if name not in run:
            raise ValueError(f"Workload '{name}' not found in capacity sensitivity run '{input_files[runs.index(run)]}'.")

h100_ordered = [h100_map[name] for name in desired_order]
ablation_data_ordered = [[run[name] for name in desired_order] for run in runs]

display_workloads = [display_name(name) for name in desired_order]

with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(display_workloads)
    writer.writerow(h100_ordered)
    for latencies in ablation_data_ordered:
        writer.writerow(latencies)

print(f"Saved {len(desired_order)} workloads to {output_file}")