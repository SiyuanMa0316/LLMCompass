import csv
import argparse
import pandas as pd
import numpy as np
parser = argparse.ArgumentParser(description="Calculate performance per area")
parser.add_argument("--input", type=str, help="Path to the input log file")
args = parser.parse_args()
# read latencies
latencies = {}
perf_per_area = {}

df = pd.read_csv(args.input)
workloads = df.columns.tolist()
# Row 0 = baseline latencies, Row 1 = new latencies
baseline = df.iloc[0].values
pim = df.iloc[1].values
proteus_data = {
"GEMM_1024x12288x12288": 95567.21/1000,
"GEMM_2048x24576x24576": 382268.84/1000,
"GEMV_1x12288x12288": 88.74/1000,
"GEMV_1x24576x24576": 177.47/1000,
"gpt3-175B_prefill": 1401086.566*96/1000,
"gpt3-175B_decode": 1306.107936*96*2048/1000,
"gpt3-6.7B_prefill": 572829.6539*32/1000,
"gpt3-6.7B_decode": 532.46928*32*2048/1000,
"Llama-3.1-70B_prefill": 1113802.157*80/1000,
"Llama-3.1-70B_decode": 1035.359808*80*2048/1000,
"Llama-3.1-8B_prefill": 556901.138*32/1000,
"Llama-3.1-8B_decode": 517.679904*32*2048/1000,
}
print(proteus_data)
proteus_latencies = [proteus_data[workload] for workload in workloads]
# print(proteus_latencies)
proteus_latencies = np.array(proteus_latencies)

pim_peripheral_area  = 5280
pim_perf_per_areas = 1E6 / pim / pim_peripheral_area

h100_perf_per_areas = 1E6 / baseline / 5740 # scaled from H100 die area and flattend HBM area, scalling to 14nm 

proteus_dram_area = 8*8/16 * 66 #Proteus is 8GB, 16Gb DRAM chip is 66mm^2
proteus_peripheral_area = proteus_dram_area * 0.01 #assume 1% area overhead (from Ambit)
print(f"Proteus peripheral area: {proteus_peripheral_area}mm^2, dram area: {proteus_dram_area}mm^2")
proteus_perf_per_areas = 1E6 / proteus_latencies / proteus_peripheral_area

# write to csv
output_file = f"perf_per_area_{args.input.split('latencies_')[1].replace('.csv', '')}.csv"
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(workloads)
    writer.writerow(h100_perf_per_areas)
    writer.writerow(pim_perf_per_areas)
    writer.writerow(proteus_perf_per_areas)
print(f"Performance per area results written to {output_file}")
