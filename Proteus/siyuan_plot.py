import matplotlib.pyplot as plt
import numpy as np

# Data
K = np.array([32, 64, 128, 256])
reuse = np.array([1056, 4160, 16512, 65792])
proteus = np.array([2048, 8192, 32768, 131072])

# Plot
plt.figure(figsize=(3.2, 2.3))  # fits 2-column paper width
plt.plot(K, reuse, marker='o', linewidth=1.5, markersize=4, label='Input/Weight Reuse')
plt.plot(K, proteus, marker='s', linewidth=1.5, markersize=4, label='Proteus')

# Axes and labels
plt.xlabel('Matrix Dimension K', fontsize=9)
plt.ylabel('Number of Vector Accesses', fontsize=9)
plt.xticks(K, fontsize=8)
plt.yticks(fontsize=8)
plt.yscale('log')  # log scale helps show growth trend
plt.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.7)
plt.legend(fontsize=8, frameon=False)

# Layout tuning
plt.tight_layout()
plt.savefig('vector_access_comparison.pdf', bbox_inches='tight')
plt.show()