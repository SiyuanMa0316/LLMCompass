import csv
import argparse
parser = argparse.ArgumentParser(description="Extract latencies from simulation output")
# parser.add_argument("--input", type=str, default="run_all_output_10-3", help="Path to the input log file")

args = parser.parse_args()

# input_file = args.input
input_files = ['run_all_output_base', "run_all_output_int4", "run_all_output_int2"]
output_file = f"latencies_sensitivity_datatype.csv"

workloads = []

ablation_data = []
h100_latencies = [0.0002655339506, 0.0020424524691358, 0.000045049255441008, 0.000180182359679266, 0.35947, 238.99, 0.02766, 27.39910, 0.13564, 102.8096, 0.02542602351, 26.05659218]
for input_file in input_files:
    with open(input_file, "r") as f:
        latencies = []
        for line in f:
            if line.startswith("simulated latency:"):
                parts = line.strip().replace("simulated latency:", "").split()
                print(parts)
                if len(parts) == 2:
                    workload = parts[0]
                    latency = parts[1]
                    workload = workload.replace("GEMM_1x", "GEMV_1x").replace("1024x12288x12288","small").replace("2048x24576x24576","large").replace("1x12288x12288","small").replace("1x24576x24576","large")
                    workloads.append(workload)
                    latencies.append(latency)
    ablation_data.append(latencies)

# write CSV: workload in header, latency in first line
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(workloads)
    writer.writerow(h100_latencies)
    for latencies in ablation_data:
        writer.writerow(latencies)

print(f"Saved {len(workloads)} workloads to {output_file}")