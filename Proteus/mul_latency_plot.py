import matplotlib.pyplot as plt
import numpy as np

# Data
precision = np.array([2, 4, 8, 16])
minimum = np.array([112, 224, 448, 896])
ours = np.array([112, 224, 448, 448*4+14*16*4+14*32])
proteus = np.array([1617, 7595, 27608, 77232])

# Create figure
fig, ax1 = plt.subplots(figsize=(3.2, 1.5))  # fits in one column
ax2 = ax1.twinx()

# Plot on both y-axes
color1 = '#1f77b4'  # blue
color2 = '#ff7f0e'  # orange
color3 = '#2ca02c'  # green
ax1.plot(precision, minimum, marker='o', linewidth=1.5, markersize=4,
         color=color1, label='Full Bit-Reuse')
ax1.set_ylim(0, 8000)
ax2.plot(precision, proteus, marker='s', linewidth=1.5, markersize=4,
         color=color2, label='No Bit-Reuse')
ax2.set_ylim(0, 80000)
ax1.plot(precision, ours, marker='^', linewidth=1.5, markersize=4,
         color=color3, label='Ours')
ax1.set_ylim(0, 8000)

# Labels
ax1.set_xlabel('Integer Multiply Precision (bits)', fontsize=9)
ax1.set_ylabel('Bit-Reuse Latency (ns)', color=color1, fontsize=5)
ax2.set_ylabel('No Bit-Reuse Latency (ns)', color=color2, fontsize=5)

# Tick parameters
ax1.tick_params(axis='y', labelcolor=color1, labelsize=6)
ax2.tick_params(axis='y', labelcolor=color2, labelsize=6)
ax1.tick_params(axis='x', labelsize=8)
plt.xticks(precision, fontsize=8)

# Grid and layout
ax1.grid(True, linestyle='--', axis='x', linewidth=0.4, alpha=0.7)
fig.tight_layout()

# Combined legend (top-center)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

fig.legend(lines1 + lines2, labels1 + labels2,
           loc='upper left', ncol=1, fontsize=6, frameon=False,
           bbox_to_anchor=(0.22, 0.9))

# Save and show
plt.savefig('integer_multiplication_latency_dual_axis_ours.pdf', bbox_inches='tight')
plt.show()
