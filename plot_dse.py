import pandas as pd

import matplotlib.pyplot as plt
import argparse
# Argument parser for command line arguments
parser = argparse.ArgumentParser(description='Plot DSE results.')
parser.add_argument('--file', type=str,
                    help='CSV file containing the DSE results')
args = parser.parse_args()
# Load CSV data
df = pd.read_csv(args.file)

# Scatter plot: total capacity vs latency
plt.scatter(df['total_capacity'], df['latency'], alpha=0.7)
#add dot at (80, 0.0002)
if "gemm" in args.file:
    plt.scatter(80, 0.0002, color='red')
elif "gemv" in args.file:
    plt.scatter(80, 0.000045, color='red')
plt.xlabel('Total Capacity')
plt.ylabel('Latency')
# Set the x-axis to logarithmic scale
plt.xscale('log')
plt.title('Scatter Plot of Capacity vs Latency')
plt.grid(True)
# plt.show()
plt.savefig(f'{args.file}.png')