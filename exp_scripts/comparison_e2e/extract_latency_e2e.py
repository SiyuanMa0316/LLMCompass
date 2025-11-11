import csv
import argparse

parser = argparse.ArgumentParser(description="Compute LLM E2E (1024->128), Decode Throughput, and Prefill Latencies")
parser.add_argument("--input", type=str, default="run_all_output_11-8", help="Path to the input log file")
parser.add_argument("--decode_tokens", type=int, default=128, help="Number of generated tokens for throughput calculation")
args = parser.parse_args()

input_file = args.input
lat_output_file = f"latencies_e2e_1024to128_{args.input}.csv"
thr_output_file = f"throughput_decode_1024to128_{args.input}.csv"
prefill_output_file = f"latencies_prefill_1024to128_{args.input}.csv"

# === Workload order ===
workload_order = [
    "GEMM_1024x12288x12288", "GEMM_2048x24576x24576",
    "GEMV_1x12288x12288", "GEMV_1x24576x24576",
    "gpt3-175B_prefill", "gpt3-175B_decode",
    "gpt3-6.7B_prefill", "gpt3-6.7B_decode",
    "Llama-3.1-70B_prefill", "Llama-3.1-70B_decode",
    "Llama-3.1-8B_prefill", "Llama-3.1-8B_decode",
]

# === Baseline latencies (H100, in seconds) ===
h100_latencies = [
    0.0002655339506, 0.0020424524691358,
    0.000045049255441008, 0.000180182359679266,
    0.35947, 238.99,
    0.02766, 27.39910,
    0.13564, 102.8096,
    0.02542602351, 26.05659218
]

# === Proteus latencies (ms → s) ===
proteus_latencies_ms = {
    "GEMM_1024x12288x12288": 95567.21,
    "GEMM_2048x24576x24576": 382268.84,
    "GEMV_1x12288x12288": 88.74,
    "GEMV_1x24576x24576": 177.47,
    "gpt3-175B_prefill": 1401086.566 * 96,
    "gpt3-175B_decode": 1306.107936 * 96 * 2048,
    "gpt3-6.7B_prefill": 572829.6539 * 32,
    "gpt3-6.7B_decode": 532.46928 * 32 * 2048,
    "Llama-3.1-70B_prefill": 1113802.157 * 80,
    "Llama-3.1-70B_decode": 1035.359808 * 80 * 2048,
    "Llama-3.1-8B_prefill": 556901.138 * 32,
    "Llama-3.1-8B_decode": 517.679904 * 32 * 2048,
}
proteus_latencies = [proteus_latencies_ms[w] / 1e3 for w in workload_order]  # s

# === Parse simulated latencies ===
sim_latencies = []
with open(input_file, "r") as f:
    for line in f:
        if line.startswith("simulated latency:"):
            parts = line.strip().replace("simulated latency:", "").split()
            if len(parts) == 2:
                sim_latencies.append(float(parts[1]))

if len(sim_latencies) < len(workload_order):
    print("Warning: fewer simulated latencies than expected.")

# === Helpers ===
def get_latency(lat_list, name):
    try:
        idx = workload_order.index(name)
        return lat_list[idx]
    except (ValueError, IndexError):
        return 0

# === 1. End-to-End Latency (prefill + scaled decode) ===
def compute_e2e(lat_list):
    e2e = []
    for prefix in ["gpt3-175B", "gpt3-6.7B", "Llama-3.1-70B", "Llama-3.1-8B"]:
        prefill = get_latency(lat_list, f"{prefix}_prefill")
        decode_full = get_latency(lat_list, f"{prefix}_decode")
        decode_scaled = decode_full * (args.decode_tokens / 2048)
        e2e.append(prefill + decode_scaled)
    return e2e

# === 2. Decode Throughput (tokens/s) ===
def compute_decode_throughput(lat_list):
    thr = []
    for prefix in ["gpt3-175B", "gpt3-6.7B", "Llama-3.1-70B", "Llama-3.1-8B"]:
        decode_full = get_latency(lat_list, f"{prefix}_decode")
        decode_scaled = decode_full * (args.decode_tokens / 2048)
        thr.append(args.decode_tokens / decode_scaled if decode_scaled > 0 else 0)
    return thr

# === 3. Prefill Latencies ===
def compute_prefill(lat_list):
    pref = []
    for prefix in ["gpt3-175B", "gpt3-6.7B", "Llama-3.1-70B", "Llama-3.1-8B"]:
        pref.append(get_latency(lat_list, f"{prefix}_prefill"))
    return pref

# === Compute all ===
h100_e2e = compute_e2e(h100_latencies)
proteus_e2e = compute_e2e(proteus_latencies)
sim_e2e = compute_e2e(sim_latencies)

h100_thr = compute_decode_throughput(h100_latencies)
proteus_thr = compute_decode_throughput(proteus_latencies)
sim_thr = compute_decode_throughput(sim_latencies)

h100_pref = compute_prefill(h100_latencies)
proteus_pref = compute_prefill(proteus_latencies)
sim_pref = compute_prefill(sim_latencies)

# === Write CSVs ===
header = ["gpt3-175B", "gpt3-6.7B", "Llama-3.1-70B", "Llama-3.1-8B"]

def write_csv(filename, *rows):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)

write_csv(lat_output_file, h100_e2e, proteus_e2e, sim_e2e)
print(f"Saved end-to-end latencies to {lat_output_file}")

write_csv(thr_output_file, h100_thr, proteus_thr, sim_thr)
print(f"Saved decode throughput to {thr_output_file}")

write_csv(prefill_output_file, h100_pref, proteus_pref, sim_pref)
print(f"Saved prefill latencies to {prefill_output_file}")
