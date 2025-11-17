import csv
import argparse
import os

BASE_INPUT_TOKENS = 1024    # baseline prefill length
BASE_OUTPUT_TOKENS = 2048   # baseline decode length

# Application-level model names and their CSV name prefixes
MODEL_NAMES = ["gpt3-175B", "Llama3-70B", "Llama3-8B","gpt3-6.7B"]

# Scenarios: (scenario_name, N_in, N_out)
SCENARIOS = [
    ("large_read",      8192,  256),   # long prompt, short answer
    ("regular_chat",    1024,  128),   # typical chat
    ("long_generation", 1024,  4096),  # long answer/story
]

def display_name(name: str) -> str:
    """Normalize model identifiers if you want nicer column names."""
    return name.replace("Llama3", "Llama3").replace("_", "-")


def parse_latency_csv(path):
    """
    Read the latency CSV.

    Returns:
        header: list of column names
        backend_names: list of backend names (Backend_1, Backend_2, ...)
        latencies_per_backend: list of dicts: [{col_name: value_in_seconds}, ...]
    """
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        raise ValueError("Input CSV is empty")

    header = [col.strip() for col in rows[0]]
    data_rows = rows[1:]

    latencies_per_backend = []
    backend_names = []
    backend_names = ["H100", "Proteus", "DREAM(Ours)"]

    for i, row in enumerate(data_rows):
        if not row:
            continue
    #     backend_name = f"Backend_{i+1}"
    #     backend_names.append(backend_name)
# Row 0 = baseline latencies, Row 1 = Proteus latencies, Row 2 = new latencies
        values = {}
        for col_name, val_str in zip(header, row):
            val_str = val_str.strip()
            if val_str == "":
                continue
            values[col_name] = float(val_str)
        latencies_per_backend.append(values)

    return header, backend_names, latencies_per_backend


def get_prefill_decode(lat_dict, model_name):
    """
    From a dict {column_name: value}, extract prefill and decode latencies
    for a given model.

    Expects column names like:
      gpt3-175B-prefill, gpt3-175B-decode, etc.
    Values are assumed to be in **seconds**.
    """
    prefill_key = f"{model_name}-prefill"
    decode_key = f"{model_name}-decode"

    prefill = lat_dict.get(prefill_key, 0.0)
    decode = lat_dict.get(decode_key, 0.0)
    return prefill, decode


def estimate_e2e_ms(lat_dict, model_name, n_in, n_out):
    """
    Estimate E2E latency in milliseconds for one backend/model/scenario.

    lat_dict: dict mapping 'model-phase' -> latency (seconds)
    model_name: e.g. "gpt3-175B"
    n_in, n_out: token counts for this scenario
    """
    prefill_base_s, decode_base_s = get_prefill_decode(lat_dict, model_name)

    if prefill_base_s <= 0.0 and decode_base_s <= 0.0:
        return 0.0

    prefill_scaled_s = prefill_base_s * (n_in / BASE_INPUT_TOKENS)
    decode_scaled_s = decode_base_s * (n_out / BASE_OUTPUT_TOKENS)
    total_s = prefill_scaled_s + decode_scaled_s

    # convert to ms
    return total_s * 1e3


def main():
    parser = argparse.ArgumentParser(
        description="Scale LLM prefill/decode latencies from a CSV (1024→2048 baseline) "
                    "to different (N_in, N_out) scenarios and output E2E latencies in ms."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input latency CSV file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to the output CSV file (optional)"
    )
    args = parser.parse_args()

    input_csv = args.input
    if args.output is None:
        base = os.path.splitext(os.path.basename(input_csv))[0]
        output_csv = f"scaled_latencies_ms.csv"
    else:
        output_csv = args.output

    header, backend_names, latencies_per_backend = parse_latency_csv(input_csv)

    # Build output table:
    # Scenario, Backend, gpt3-175B, gpt3-6.7B, Llama3-70B, Llama3-8B
    out_header = ["Scenario", "Backend"] + [display_name(m) for m in MODEL_NAMES]
    rows = []

    for scenario_name, n_in, n_out in SCENARIOS:
        for backend_name, lat_dict in zip(backend_names, latencies_per_backend):
            model_values_ms = [
                estimate_e2e_ms(lat_dict, model_name, n_in, n_out)
                for model_name in MODEL_NAMES
            ]
            rows.append([scenario_name, backend_name] + model_values_ms)

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(out_header)
        writer.writerows(rows)

    print(f"Saved scaled E2E latencies (ms) to {output_csv}")


if __name__ == "__main__":
    main()
