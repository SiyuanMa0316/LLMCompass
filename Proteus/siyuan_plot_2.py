import matplotlib.pyplot as plt
import numpy as np

# Data
precision = np.array([2, 4, 8, 16])
minimum = np.array([112, 224, 448, 896])
proteus = np.array([1617, 7595, 27608, 77232])

# Plot
plt.figure(figsize=(3.2, 1.5))  # fits 2-column paper width
plt.plot(precision, minimum, marker='o', linewidth=1.5, markersize=4, label='Full Bit-Reuse')
plt.plot(precision, proteus, marker='s', linewidth=1.5, markersize=4, label='No Bit-Reuse')

# Axes and labels
plt.xlabel('Integer Multiply Precision (bits)', fontsize=9)
plt.ylabel('Latency (ns)', fontsize=9)
plt.xticks(precision, fontsize=8)
plt.yticks(fontsize=8)
# plt.yscale('log')  # logarithmic for clear growth visualization
plt.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.7)
plt.legend(fontsize=8, frameon=False)

# Layout tuning
plt.tight_layout()
plt.savefig('integer_multiplication_latency_comparison.pdf', bbox_inches='tight')
plt.show()
