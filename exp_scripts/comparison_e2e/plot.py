import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Configuration
BASE_FONT = 7
plt.rcParams.update(
	{
		"font.size": BASE_FONT,
		"axes.labelsize": BASE_FONT + 1,
		"axes.titlesize": BASE_FONT,
		"legend.fontsize": BASE_FONT,
		"xtick.labelsize": BASE_FONT,
		"ytick.labelsize": BASE_FONT,
		"figure.dpi": 300,
	}
)

FIG_WIDTH = 2.7
FIGSIZE = (FIG_WIDTH, 2)
BAR_LINEWIDTH = 0.3
BAR_WIDTH = 0.2
AXHLINE_WIDTH = 0.3
SPINE_LINEWIDTH = 0.8
ANNOT_FONTSIZE = BASE_FONT - 1


# Scenario-specific color combinations
SCENARIO_COLORS = {
	"large_read": ["#CC9892", "#F7BDA9", "#ED7249"],       # Teal, Gold, Rose
	# "regular_chat": ["#7EA6D9", "#90CED7", "#7ECA86"],     # Sky blue, Orange, Mint
	"long_generation": ["#5BB5AC", "#D8B365", "#DE526C"],  # Apricot, Taupe, Brown
}

SCENARIOS_INPUT_OUTPUT_TOKENS = [
	("large_read",      8192,  256),   # long prompt, short answer
	# ("regular_chat",    1024,  128),   # typical chat
	("long_generation", 1024,  4096),  # long answer/story
]

def geom_mean(values: np.ndarray) -> float:
	values = np.asarray(values, dtype=float)
	if np.any(values <= 0):
		raise ValueError("Geometric mean requires strictly positive values.")
	return float(np.exp(np.mean(np.log(values))))

CSV_FILE = "scaled_latencies_ms.csv"
OUTPUT_DIR = Path("comparison_e2e_figures")
BASELINE_BACKEND = "H100"
SCENARIOS = [item[0] for item in SCENARIOS_INPUT_OUTPUT_TOKENS]

# Create lookup for scenario details
SCENARIO_DETAILS = {name: (inp, out) for name, inp, out in SCENARIOS_INPUT_OUTPUT_TOKENS}

# Read CSV
scenario_map = {}
model_labels = None

with open(CSV_FILE, "r", newline="") as f:
	reader = csv.DictReader(f)
	fieldnames = reader.fieldnames
	model_labels = [name for name in fieldnames if name not in {"Scenario", "Backend"}]
	
	for row in reader:
		scenario = row["Scenario"].strip()
		if scenario not in SCENARIOS:
			continue
		backend = row["Backend"].strip()
		values = np.array([float(row[label]) for label in model_labels], dtype=float)
		scenario_map.setdefault(scenario, {})[backend] = values

# Create figures for each scenario
figs = []
for scenario in SCENARIOS:
	scenario_values = scenario_map[scenario]
	
	# Normalize to H100 baseline and convert to throughput (1/latency)
	baseline = scenario_values[BASELINE_BACKEND]
	normalized_throughput = {backend: baseline / values for backend, values in scenario_values.items()}
	
	# Compute geomean for each backend
	geomean_values = {backend: geom_mean(values) for backend, values in normalized_throughput.items()}
	
	# Append geomean to model_labels and data
	labels_with_geomean = list(model_labels) + ["Geomean"]
	normalized_throughput_with_geomean = {
		backend: np.append(values, geomean_values[backend]) 
		for backend, values in normalized_throughput.items()
	}
	
	# Plot
	backend_order = [BASELINE_BACKEND] + [b for b in normalized_throughput_with_geomean if b != BASELINE_BACKEND]
	
	fig, ax = plt.subplots(figsize=FIGSIZE)
	for spine in ax.spines.values():
		spine.set_linewidth(SPINE_LINEWIDTH)
	
	x = np.arange(len(labels_with_geomean))
	n_series = len(backend_order)
	offsets = (np.arange(n_series) - (n_series - 1) / 2.0) * BAR_WIDTH
	
	for idx, backend in enumerate(backend_order):
		values = normalized_throughput_with_geomean[backend]
		# Use scenario-specific colors
		scenario_colors = SCENARIO_COLORS.get(scenario, ["#5BB5AC", "#D8B365", "#DE526C"])
		color = scenario_colors[idx % len(scenario_colors)]
		ax.bar(
			x + offsets[idx],
			values,
			width=BAR_WIDTH,
			label=backend,
			color=color,
			linewidth=BAR_LINEWIDTH,
			edgecolor="black",
			alpha=1,
		)
	
	ax.set_xticks(x)
	ax.set_xticklabels(labels_with_geomean, rotation=10, ha="center", fontsize=plt.rcParams["xtick.labelsize"])
	if scenario != "large_read":
		ax.set_ylabel("Normalized throughput")
	ax.set_yscale("log")
	
	# Get scenario details for annotation
	input_tokens, output_tokens = SCENARIO_DETAILS[scenario]
	scenario_title = scenario.replace("_", " ").title()
	scenario_annotation = f"{scenario_title}: {input_tokens} input tokens, {output_tokens} output tokens"
	
	# Set title with scenario annotation at top
	# ax.set_title(scenario_annotation, loc='center', pad=15)
	
	ax.axhline(y=1.0, color="black", linestyle="--", linewidth=AXHLINE_WIDTH, alpha=0.7)
	
	# Balance y-axis limits for better visualization
	ymin = ax.get_ylim()[0]
	ymax = ax.get_ylim()[1] * 10
	ax.set_ylim(ymin, ymax)
	
	legend = ax.legend(
		frameon=False,
		ncol=n_series,
		loc="upper center",
		bbox_to_anchor=(0.5, 1.03),
		handlelength=1.2,
		columnspacing=0.8,
	)
	legend.set_title(None)
	
	# Add bar value annotations (only for values > 1)
	for backend_idx, backend in enumerate(backend_order):
		values = normalized_throughput_with_geomean[backend]
		for x_idx, value in enumerate(values):
			if value > 1.0:  # Only annotate if value > 1
				bar_x = x_idx + offsets[backend_idx]
				height = value
				ax.text(
					bar_x,
					height * 1.05,
					f"x{value:.1f}",
					ha="center",
					va="bottom",
					fontsize=ANNOT_FONTSIZE,
				)
	
	# Add backend annotation label at bottom
	# fig.text(0.5, 0.02, "Backends", ha='center', fontsize=plt.rcParams["font.size"], 
	#          transform=fig.transFigure, style='italic', color='gray')
	
	plt.tight_layout(pad=0.35)
	
	# Save figure
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	fig_path = OUTPUT_DIR / f"{scenario}_normalized_throughput.pdf"
	fig.savefig(fig_path, bbox_inches="tight", pad_inches=0.02)
	print(f"Saved figure -> {fig_path}")
	
	figs.append(fig)

# Show all figures at the same time
plt.show()
