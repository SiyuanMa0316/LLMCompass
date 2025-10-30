import subprocess
# import argparse
import csv
# parser = argparse.ArgumentParser(description="Automate running comparison experiments")
# parser.add_argument("--input", type=str, help="Path to the input log file")
# args = parser.parse_args()

#extract latencies
result_shell = subprocess.run(f"python extract_latency.py", shell=True, capture_output=True, text=True)
print(result_shell.stdout)

#replace GEMM_1 with GEMV_1 in the first row of the csv
csv_file = f"latencies_ablation.csv"
rows = []
with open(csv_file, "r", newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        rows.append([cell.replace("GEMM_1x", "GEMV_1x").replace("1024x12288x12288","small").replace("2048x24576x24576","large").replace("1x12288x12288","small").replace("1x24576x24576","large") for cell in row])
with open(csv_file, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerows(rows)

#plot speedup
result_shell = subprocess.run(f"python plot.py latencies_ablation.csv", shell=True, capture_output=True, text=True)

print(result_shell.stdout)