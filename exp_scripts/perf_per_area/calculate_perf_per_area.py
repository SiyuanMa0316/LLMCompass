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
print(workloads)
# Row 0 = baseline latencies, Row 1 = new latencies
baseline = df.iloc[0].values
pim = df.iloc[2].values
# print(proteus_data)
proteus_latencies = df.iloc[1].values
# print(proteus_latencies)
# proteus_latencies = np.array(proteus_latencies)

pim_peripheral_area  = 5280
pim_perf_per_areas = 1E6 / pim / pim_peripheral_area

h100_perf_per_areas = 1E6 / baseline / 5740 # scaled from H100 die area and flattend HBM area, scalling to 14nm 

proteus_dram_area = 8*8/16 * 66 #Proteus is 8GB, 16Gb DRAM chip is 66mm^2
proteus_peripheral_area = proteus_dram_area * 0.01 #assume 1% area overhead (from Ambit)
print(f"Proteus peripheral area: {proteus_peripheral_area}mm^2, dram area: {proteus_dram_area}mm^2")
proteus_perf_per_areas = 1E6 / proteus_latencies / proteus_peripheral_area

# write to csv
output_file = f"perf_per_area_{args.input.split('latencies_')[1].replace('.csv', '')}.csv"
print(f"Writing performance per area to {output_file}")
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(workloads)
    print(workloads)
    writer.writerow(h100_perf_per_areas) 
    writer.writerow(proteus_perf_per_areas)
    writer.writerow(pim_perf_per_areas)
   
print(f"Performance per area results written to {output_file}")
