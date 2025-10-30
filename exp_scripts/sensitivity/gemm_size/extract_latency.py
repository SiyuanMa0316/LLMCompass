import csv
import argparse
parser = argparse.ArgumentParser(description="Extract latencies from simulation output")
parser.add_argument("--input", type=str, default="run_all_output_10-3", help="Path to the input log file")

args = parser.parse_args()

input_file = args.input
output_file = f"latencies_{args.input}.csv"

workloads = []
latencies = []
# h100_latencies = [0.0002655339506, 0.0020424524691358, 0.000045049255441008, 0.000180182359679266, 0.35947, 238.99, 0.02766, 27.39910, 0.13564, 102.8096, 0.02542602351, 26.05659218]

with open(input_file, "r") as f:
    for line in f:
        if line.startswith("simulated latency:"):
            parts = line.strip().replace("simulated latency:", "").split()
            print(parts)
            if len(parts) == 2:
                workload = parts[0]
                latency = parts[1]
                workloads.append(workload)
                latencies.append(latency)

# proteus_latencies = {
# "GEMM_1024x12288x12288": 95567.21/1000,
# "GEMM_2048x24576x24576": 382268.84/1000,
# "GEMV_1x12288x12288": 88.74/1000,
# "GEMV_1x24576x24576": 177.47/1000,
# "gpt3-175B_prefill": 1401086.566*96/1000,
# "gpt3-175B_decode": 1306.107936*96*2048/1000,
# "gpt3-6.7B_prefill": 572829.6539*32/1000,
# "gpt3-6.7B_decode": 532.46928*32*2048/1000,
# "Llama-3.1-70B_prefill": 1113802.157*80/1000,
# "Llama-3.1-70B_decode": 1035.359808*80*2048/1000,
# "Llama-3.1-8B_prefill": 556901.138*32/1000,
# "Llama-3.1-8B_decode": 517.679904*32*2048/1000,
# }
# proteus_latencies = proteus_latencies.values()

# write CSV: workload in header, latency in first line
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(workloads)
    # writer.writerow(h100_latencies)
    # writer.writerow(proteus_latencies)
    writer.writerow(latencies)

print(f"Saved {len(workloads)} workloads to {output_file}")