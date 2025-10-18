from design_space_exploration.dse import template_to_system, read_architecture_template
from software_model.utils import data_type_dict, Tensor
from software_model.transformer_hyper import Transformer_hyper
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import copy
import csv
import os

csv_header = [
    "phase", "model_name", "layers", "seq_len", "output_len",
    "x", "rank_count", "row", "col", "arr_count", "device_count", "channel_count",
    "latency", "total_latency", "capacity_GB"
]

def run_prefill(system, llm_hyper, precision="int8", bs=1, seq_len=1024):
    if "gpt" in llm_hyper.name:
        from software_model.transformer import TransformerBlockInitComputationTP
        model_prefill = TransformerBlockInitComputationTP(
            model_name=llm_hyper.name,
            d_model=llm_hyper.d_model,
            n_heads=llm_hyper.num_heads,
            device_count=1,
            data_type=data_type_dict[precision],
        )
    elif "Llama" in llm_hyper.name:
        from software_model.transformer_Llama import TransformerBlockInitComputationTP
        model_prefill = TransformerBlockInitComputationTP(
            model_name=llm_hyper.name,
            d_model=llm_hyper.d_model,
            n_kv_heads=llm_hyper.num_kv_heads,
            n_heads=llm_hyper.num_heads,
            ffn_dim=llm_hyper.ffn_dim,
            device_count=1,
            data_type=data_type_dict[precision],
        )
    else:
        raise ValueError("Unsupported model type")

    _ = model_prefill(Tensor([bs, seq_len, llm_hyper.d_model], data_type_dict[precision]))
    latency = model_prefill.compile_and_simulate(system, compile_mode="specific")
    return latency, latency * llm_hyper.num_layers


def run_decode(system, llm_hyper, pipelined=False, precision="int8", bs=1, seq_len=1024, output_len=2048):
    if pipelined:
        system.device.compute_module.channel_count = 1
        system.device.compute_module.rank_count = 1

    if "gpt" in llm_hyper.name:
        from software_model.transformer import TransformerBlockAutoRegressionTP
        model_decode = TransformerBlockAutoRegressionTP(
            model_name=llm_hyper.name,
            d_model=llm_hyper.d_model,
            n_heads=llm_hyper.num_heads,
            device_count=1,
            data_type=data_type_dict[precision],
        )
    elif "Llama" in llm_hyper.name:
        from software_model.transformer_Llama import TransformerBlockAutoRegressionTP
        model_decode = TransformerBlockAutoRegressionTP(
            model_name=llm_hyper.name,
            d_model=llm_hyper.d_model,
            n_kv_heads=llm_hyper.num_kv_heads,
            n_heads=llm_hyper.num_heads,
            ffn_dim=llm_hyper.ffn_dim,
            device_count=1,
            data_type=data_type_dict[precision],
        )
    else:
        raise ValueError("Unsupported model type")

    _ = model_decode(Tensor([bs, 1, llm_hyper.d_model], data_type_dict[precision]), seq_len)
    latency = model_decode.compile_and_simulate(system, compile_mode="specific")

    if pipelined:
        total = latency * llm_hyper.num_layers + latency * (output_len - 1)
    else:
        total = latency * llm_hyper.num_layers * output_len
    return latency, total


def run_simulation(x, rank_count, col_count, arr_count, device_count, row_count,
                   base_specs, llm_hyper, prefill=True, decode=True, pipelined=False):

    specs = copy.deepcopy(base_specs)
    if x == "x4":
        specs["compute_module"]['bank']['device_data_width'] = 4
        bank_count = 32
    elif x == "x8":
        specs["compute_module"]['bank']['device_data_width'] = 8
        bank_count = 32
    elif x == "x16":
        specs["compute_module"]['bank']['device_data_width'] = 16
        bank_count = 16
    else:
        raise ValueError("Unsupported x")

    specs["compute_module"]["rank_count"] = rank_count
    specs["compute_module"]["bank"]["array_cols"] = col_count
    specs["compute_module"]["bank"]["array_rows"] = row_count
    specs["compute_module"]["bank"]["array_count"] = arr_count
    specs["compute_module"]["bank"]["device_count"] = device_count
    specs["compute_module"]["bank_count"] = bank_count
    specs["compute_module"]["channel_count"] = 10

    system = template_to_system(specs)
    simdram = system.device

    results = []
    if prefill:
        lat, total = run_prefill(system, llm_hyper)
        results.append([
            "prefill", llm_hyper.name, llm_hyper.num_layers, 1024, "-", x,
            rank_count, row_count, col_count, arr_count, device_count, 10,
            lat, total, simdram.compute_module.capacity / 1024 / 1024 / 1024
        ])
    if decode:
        lat, total = run_decode(system, llm_hyper, pipelined=pipelined)
        results.append([
            "decode", llm_hyper.name, llm_hyper.num_layers, 1024, 2048, x,
            rank_count, row_count, col_count, arr_count, device_count, 10,
            lat, total, simdram.compute_module.capacity / 1024 / 1024 / 1024
        ])
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DSE for GPT-3/LLaMA on DRAM PIM")
    parser.add_argument("--config", type=str, required=True, help="Base config file")
    parser.add_argument("--model", type=str, required=True, help="Model hyperparameter file")
    parser.add_argument("--prefill", action="store_true", help="Run prefill")
    parser.add_argument("--decode", action="store_true", help="Run decode")
    parser.add_argument("--pipelined", action="store_true", help="Run pipelined decode")
    parser.add_argument("--dse", action="store_true", help="Run design space exploration")
    args = parser.parse_args()

    base_specs = read_architecture_template(args.config)
    llm_hyper = Transformer_hyper()
    llm_hyper.read_from_json(args.model)

    # DSE parameter ranges
    row_count_range = [16*2**10, 32*2**10, 64*2**10]
    col_arr_count_x4 = [(256, 32), (512, 16)]
    col_arr_count_x8 = [(256, 64), (512, 32)]
    col_arr_count_x16 = [(256, 64), (512, 32)]
    device_count_x4 = [16, 32]
    device_count_x8 = [8, 16]
    device_count_x16 = [4, 8]
    rank_count_range = [1, 2, 4, 8, 16]

    csv_data = []
    futures = []

    if not args.dse:
        results = run_simulation("x8", 8, 256, 64, 8, 16*2**10,
                                 base_specs, llm_hyper,
                                 prefill=args.prefill, decode=args.decode, pipelined=args.pipelined)
        csv_data.extend(results)
    else:
        with ProcessPoolExecutor(max_workers=8) as executor:
            for x in ["x4", "x8", "x16"]:
                if x == "x4":
                    col_arr_range = col_arr_count_x4
                    dev_range = device_count_x4
                elif x == "x8":
                    col_arr_range = col_arr_count_x8
                    dev_range = device_count_x8
                else:
                    col_arr_range = col_arr_count_x16
                    dev_range = device_count_x16

                for rank_count in rank_count_range:
                    for col_count, arr_count in col_arr_range:
                        for device_count in dev_range:
                            for row_count in row_count_range:
                                futures.append(
                                    executor.submit(
                                        run_simulation,
                                        x, rank_count, col_count, arr_count, device_count, row_count,
                                        base_specs, llm_hyper,
                                        args.prefill, args.decode, args.pipelined
                                    )
                                )

            for future in as_completed(futures):
                try:
                    res = future.result()
                    csv_data.extend(res)
                except Exception as e:
                    print(f"Simulation failed: {e}")

    filename = f"llm_dse_{llm_hyper.name}_{'prefill' if args.prefill else ''}{'decode' if args.decode else ''}.csv"
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        writer.writerows(csv_data)
    print(f"Results written to {filename}")
