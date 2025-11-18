import matplotlib.pyplot as plt
import squarify
import numpy as np
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description="Plot area breakdown")
parser.add_argument("--show-gpu", action="store_true", default=False, 
                    help="Show GPU comparison plots (default: False)")
args = parser.parse_args()

# SIMDRAM area breakdown data
pnr_factor = 1.64
tech_node_factor = (14/45) ** 2 # assuming area scales with square of tech node
area_data = {
    "bank broadcast": 3.383552 * pnr_factor * tech_node_factor ,
    "rank broadcast": 0.21476288 * pnr_factor * tech_node_factor,
    "array broadcast": 218.99837440000002 * pnr_factor * tech_node_factor,
    "popcount reduction": 3620.732928 * pnr_factor * tech_node_factor,
    "bit-serial PE": 4496.293888 * pnr_factor * tech_node_factor,
    "locality buffer": 337.691803648 * pnr_factor * tech_node_factor,
}

# GPU metrics
hbm_area = 4078 
gpu_compute_die_area = 3256

# Calculate totals
peripheral_area_total = sum(area_data.values())
total_gpu_area = hbm_area + gpu_compute_die_area

# Calculate percentages
peripheral_percentage = (peripheral_area_total / total_gpu_area) * 100
gpu_percentage = 100 - peripheral_percentage

# Sort by area (largest first for better visualization)
sorted_items = sorted(area_data.items(), key=lambda x: x[1], reverse=True)
labels = [item[0] for item in sorted_items]
sizes = [item[1] for item in sorted_items]

# Calculate total area
total_area = sum(sizes)

# Separate into major (>1%) and minor (<1%) components for better visualization
major_threshold = total_area * 0.01
major_items = [(label, size) for label, size in sorted_items if size >= major_threshold]
minor_items = [(label, size) for label, size in sorted_items if size < major_threshold]

# Define colors for each component
color_map = {
    "bank broadcast": "#FF6B6B",
    "rank broadcast": "#FFA500",
    "array broadcast": "#4ECDC4",
    "popcount reduction": "#95E1D3",
    "bit-serial PE": "#6C5CE7",
    "locality buffer": "#A29BFE",
}

# Create figure with 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))
major_labels = [item[0] for item in major_items]
major_sizes = [item[1] for item in major_items]
major_colors = [color_map[label] for label in major_labels]

major_labels_with_info = [
    f"{label}\n{(size/total_area*100):.1f}%"
    for label, size in major_items
]

squarify.plot(
    sizes=major_sizes,
    label=major_labels_with_info,
    color=major_colors,
    ax=ax1,
    text_kwargs={"fontsize": 11, "weight": "bold"},
    edgecolor="white",
    linewidth=2,
)

ax1.set_title("Area Breakdown (Major Components >1%)", 
             fontsize=12, weight="bold", pad=10)
ax1.axis("off")

# ===== RIGHT: Bar chart showing all components with linear scale =====
all_labels = [item[0] for item in sorted_items]
all_sizes = [item[1] for item in sorted_items]
all_percentages = [(size/total_area*100) for size in all_sizes]
all_colors = [color_map[label] for label in all_labels]

bars = ax2.barh(all_labels, all_percentages, color=all_colors, edgecolor="black", linewidth=1.5)

# Add percentage labels on bars
for i, (bar, pct, size) in enumerate(zip(bars, all_percentages, all_sizes)):
    label_text = f"{pct:.2f}%"
    
    # Place labels inside the bar for large components
    if all_labels[i] in ["bit-serial PE", "popcount reduction"]:
        x_pos = bar.get_width() * 0.5  # Middle of the bar
        ha = "center"
        va = "center"
    else:
        # Place labels outside the bar for smaller components
        x_pos = bar.get_width() + 0.5
        ha = "left"
        va = "center"
    
    ax2.text(x_pos, bar.get_y() + bar.get_height()/2, label_text,
             ha=ha, va=va, fontsize=9, weight="bold")

ax2.set_xlabel("Percentage of Total Area (%)", fontsize=11, weight="bold")
ax2.set_title("Area Breakdown (All Components - Linear Scale)", 
             fontsize=12, weight="bold", pad=10)
ax2.set_xlim(0, max(all_percentages) + 15)
ax2.grid(axis="x", alpha=0.3, linestyle="--")

# ===== MIDDLE: Pie chart showing all components =====
pie_labels = [
    f"{label}\n{(size/total_area*100):.2f}%"
    for label, size in zip(all_labels, all_sizes)
]

# Prepare labels: inside for large slices, outside for small slices
pie_labels_display = []
for label, size in zip(all_labels, all_sizes):
    percentage = (size / total_area) * 100
    if percentage >= 5:
        # Large slices: label inside
        pie_labels_display.append(f"{label}\n{percentage:.2f}%")
    else:
        # Small slices: label outside (will add annotations later)
        pie_labels_display.append("")

wedges, texts, autotexts = ax3.pie(
    all_sizes,
    labels=pie_labels_display,
    colors=all_colors,
    autopct='',
    startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 2},
    textprops={"fontsize": 9, "weight": "bold"},
)

# Add external labels with arrows for small slices (<5%)
small_slice_indices = [i for i, (label, size) in enumerate(zip(all_labels, all_sizes)) 
                       if (size / total_area) * 100 < 5]

for idx in small_slice_indices:
    wedge = wedges[idx]
    angle = (wedge.theta2 + wedge.theta1) / 2
    # Convert to radians
    angle_rad = np.radians(angle)
    
    # Position for label (outside the pie)
    r_label = 1.3
    x_label = r_label * np.cos(angle_rad)
    y_label = r_label * np.sin(angle_rad)
    
    # Position for arrow endpoint (on the pie edge)
    r_arrow = 1.0
    x_arrow = r_arrow * np.cos(angle_rad)
    y_arrow = r_arrow * np.sin(angle_rad)
    
    percentage = (all_sizes[idx] / total_area) * 100
    label_text = f"{all_labels[idx]}\n{percentage:.2f}%"
    
    # Determine alignment
    ha = "left" if x_label > 0 else "right"
    
    # Add arrow annotation
    ax3.annotate(
        label_text,
        xy=(x_arrow, y_arrow),
        xytext=(x_label, y_label),
        fontsize=8,
        ha=ha,
        va="center",
        arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=0.2", lw=1, color="black"),
    )

ax3.set_title("Area Breakdown (Pie Chart)", 
             fontsize=12, weight="bold", pad=10)
ax3.axis("off")

plt.tight_layout()
plt.savefig("simdram_area_treemap.pdf", dpi=300, bbox_inches="tight")
plt.show()

# ===== COMPARISON: Peripheral vs GPU Area (if --show-gpu flag is set) =====
if args.show_gpu:
    fig2, ax_pie = plt.subplots(figsize=(10, 8))

    # ===== Pie chart for peripheral area breakdown =====
    # Filter components >= 1% of peripheral area
    peripheral_threshold = peripheral_area_total * 0.01
    significant_peripheral = [(label, size) for label, size in sorted_items 
                             if size >= peripheral_threshold]
    
    pie_labels_list = [item[0] for item in significant_peripheral]
    pie_sizes_list = [item[1] for item in significant_peripheral]
    pie_colors_list = [color_map[label] for label in pie_labels_list]
    
    # Prepare labels for pie chart
    pie_labels_display = []
    for label, size in significant_peripheral:
        percentage = (size / peripheral_area_total) * 100
        
        # Inside labels for large components
        if label in ["bit-serial PE", "popcount reduction"]:
            pie_labels_display.append(f"{label}\n{percentage:.1f}%")
        else:
            # Outside labels (will add annotations)
            pie_labels_display.append("")
    
    wedges_pie, texts_pie, autotexts_pie = ax_pie.pie(
        pie_sizes_list,
        labels=pie_labels_display,
        colors=pie_colors_list,
        autopct='',
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 11, "weight": "bold"},
    )
    
    # Add external labels with arrows for components needing outside labels
    for idx, (label, size) in enumerate(significant_peripheral):
        percentage = (size / peripheral_area_total) * 100
        
        # Only add external labels for locality buffer and array broadcast
        if label in ["locality buffer", "array broadcast"]:
            wedge = wedges_pie[idx]
            angle = (wedge.theta2 + wedge.theta1) / 2
            angle_rad = np.radians(angle)
            
            # Determine which side to place the label
            if label == "locality buffer":
                # Place on the right side
                r_label = 1.4
                x_label = r_label * np.cos(angle_rad)
                y_label = r_label * np.sin(angle_rad)
                ha = "left"
            else:  # array broadcast
                # Place on the left side
                r_label = 1.4
                x_label = r_label * np.cos(angle_rad)
                y_label = r_label * np.sin(angle_rad)
                ha = "right"
            
            # Position for arrow endpoint (on the pie edge)
            r_arrow = 1.0
            x_arrow = r_arrow * np.cos(angle_rad)
            y_arrow = r_arrow * np.sin(angle_rad)
            
            label_text = f"{label}\n{percentage:.1f}%"
            
            # Add arrow annotation
            ax_pie.annotate(
                label_text,
                xy=(x_arrow, y_arrow),
                xytext=(x_label, y_label),
                fontsize=10,
                ha=ha,
                va="center",
                weight="bold",
                arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=0.3", lw=1.5, color="black"),
            )
    
    ax_pie.set_title("Peripheral Area Breakdown (Components ≥1%)", 
                     fontsize=14, weight="bold", pad=20)
    ax_pie.axis("off")

    plt.tight_layout()
    plt.savefig("gpu_area_comparison.pdf", dpi=300, bbox_inches="tight")
    plt.show()

# Also print summary statistics
print("=" * 60)
print("SIMDRAM Area Breakdown Summary")
print("=" * 60)
for label, size in sorted_items:
    percentage = (size / total_area) * 100
    print(f"{label:20s}: {size:12.2f} mm² ({percentage:6.2f}%)")
print("-" * 60)
print(f"{'Total Peripheral':20s}: {peripheral_area_total:12.2f} mm²")
print("=" * 60)

if args.show_gpu:
    print("\n" + "=" * 60)
    print("GPU vs Peripheral Area Comparison")
    print("=" * 60)
    print(f"{'HBM Area':20s}: {hbm_area:12.2f} mm² ({(hbm_area/total_gpu_area*100):6.2f}%)")
    print(f"{'GPU Compute Die':20s}: {gpu_compute_die_area:12.2f} mm² ({(gpu_compute_die_area/total_gpu_area*100):6.2f}%)")
    print(f"{'Peripheral Area':20s}: {peripheral_area_total:12.2f} mm² ({peripheral_percentage:6.2f}%)")
    print("-" * 60)
    print(f"{'Total GPU Area':20s}: {total_gpu_area:12.2f} mm²")
    print("=" * 60)
    print(f"\n*** Peripheral area is {peripheral_percentage:.2f}% of total GPU area ***")
