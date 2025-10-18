import subprocess

# python calculate_perf_per_area.py --input ../comparison/latencies_run_all_output_10-3.csv 
result_shell = subprocess.run(f"python calculate_perf_per_area.py --input ../comparison/latencies_run_all_output_10-9.csv", shell=True, capture_output=True, text=True)
print(result_shell.stdout)

result_shell = subprocess.run(f"python plot.py --input perf_per_area_run_all_output_10-9.csv", shell=True, capture_output=True, text=True)
print(result_shell.stdout)