import subprocess
import argparse
import csv
parser = argparse.ArgumentParser(description="Automate running comparison experiments")
parser.add_argument("--input", type=str, help="Path to the input log file")
args = parser.parse_args()

#extract latencies
result_shell = subprocess.run(f"python extract_latency_e2e.py --input {args.input}", shell=True, capture_output=True, text=True)
print(result_shell.stdout)
#plot speedup
result_shell = subprocess.run(f"python plot_e2e_single_axis.py --throughput throughput_decode_1024to128_{args.input}.csv --prefill latencies_prefill_1024to128_{args.input}.csv", shell=True, capture_output=True, text=True)

print(result_shell.stdout)