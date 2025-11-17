import subprocess
import argparse
import csv
# python calculate_perf_per_area.py --input ../comparison/latencies_run_all_output_10-3.csv 
result_shell = subprocess.run(f"python calculate_perf_per_area.py --input ../comparison/latencies_run_all_output.csv", shell=True, capture_output=True, text=True)
print(result_shell.stdout)

#replace GEMM_1 with GEMV_1 in the first row of the csv
csv_file = f"latencies_run_all_output.csv"
rows = []
with open(csv_file, "r", newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        rows.append([cell.replace("GEMM_1x", "GEMV_1x").replace("1024x12288x12288","small").replace("2048x24576x24576","large").replace("1x12288x12288","small").replace("1x24576x24576","large") for cell in row])
with open(csv_file, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerows(rows)

result_shell = subprocess.run(f"python plot_simple.py --input perf_per_area_run_all_output.csv", shell=True, capture_output=True, text=True)
print(result_shell.stdout)