from software_model.matmul import Matmul
from software_model.utils import data_type_dict, Tensor
from software_model.strategy import Strategy
from design_space_exploration.dse import template_to_system, read_architecture_template
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import argparse
parser = argparse.ArgumentParser(description="Test GEMM on DRAM PIM")
parser.add_argument("--config", type=str, default="configs/ddr5.json", help="Path to the config file")
parser.add_argument("--dse", action='store_true', help="Run DSE. This will overwrite --config settings.")
parser.add_argument("--M", type=int, default=1024, help="Matrix M dimension")
parser.add_argument("--K", type=int, default=12288, help="Matrix K dimension")
parser.add_argument("--N", type=int, default=12288, help="Matrix N dimension")
parser.add_argument("--log_strategy", type=str, default=None, help="Path to the log strategy file")
parser.add_argument("--use_strategy", type=str, default=None, help="Path to the strategy file to use")
args = parser.parse_args()
M = args.M
K = args.K
N = args.N
model = Matmul(data_type=data_type_dict["int8"])
_ = model(
    Tensor([M, K], data_type_dict["int8"]),
    Tensor([K, N], data_type_dict["int8"]),
)

base_specs = read_architecture_template(args.config)

row_count_range = [16*2**10, 32*2**10, 64*2**10, 128*2**10]  # 16K, 32K, 64K, 128K
col_arr_count_x8 = [(256, 64), (512, 32), (1024, 16)]
col_arr_count_x4 = [(256, 32), (512, 16), (1024, 8)]
col_arr_count_x16 = [(256, 64), (512, 32), (1024, 16)]
bank_count_x8 = 32
bank_count_x4 = 32
bank_count_x16 = 16
device_count_x8 = [8, 16]
device_count_x4 = [16, 32]
device_count_x16 = [4, 8]
rank_count_range = [1, 2, 4, 8, 16, 32, 64]
channel_count = 10

csv_header = [
    "x", "parallelism", "row", "col", "tile_mapping", "arr_mapping", "latency",
    "tiling_utilization", "col_utilization",
    "capacity_utilization", "total_capacity"
]

def run_simulation(x, rank_count, col_count, arr_count, device_count, row_count):
    # Re-create model inside each process (can’t share objects safely)
    local_model = Matmul(data_type=data_type_dict["int8"])
    _ = local_model(
        Tensor([M, K], data_type_dict["int8"]),
        Tensor([K, N], data_type_dict["int8"]),
    )

    specs = copy.deepcopy(base_specs)

    if x == "x4":
        bank_count = bank_count_x4
        specs["compute_module"]['bank']['device_data_width'] = 4
    elif x == "x8":
        bank_count = bank_count_x8
        specs["compute_module"]['bank']['device_data_width'] = 8
    elif x == "x16":
        bank_count = bank_count_x16
        specs["compute_module"]['bank']['device_data_width'] = 16

    specs["compute_module"]["rank_count"] = rank_count
    specs["compute_module"]["bank"]["array_cols"] = col_count
    specs["compute_module"]["bank"]["array_rows"] = row_count
    specs["compute_module"]["bank_count"] = bank_count
    specs["compute_module"]["bank"]["array_count"] = arr_count
    specs["compute_module"]["bank"]["device_count"] = device_count
    specs["compute_module"]["channel_count"] = channel_count

    system = template_to_system(specs)
    simdram = system.device

    print(f"[PID {os.getpid()}] {simdram.info()}")
    strategy = local_model.find_simdram_mapping(simdram, debug=False)
    latency = local_model.compile_and_simulate(
        simdram, compile_mode="specific", strategy=strategy, debug=True
    )

    return [
        x,
        simdram.compute_module.parallelisms,
        simdram.compute_module.bank.arr_rows,
        simdram.compute_module.bank.arr_cols,
        local_model.stats.strategy.tile_mapping,
        local_model.stats.strategy.arr_mapping,
        latency,
        local_model.stats.tiling_utilization,
        local_model.stats.simd_utilization,
        local_model.stats.capacity_utilization,
        simdram.compute_module.capacity / 1024 / 1024 / 1024,  # in GB
    ]

csv_data = []
futures = []

if __name__ == "__main__":  # required for Windows
    import os
    with ProcessPoolExecutor(max_workers=8) as executor:  # tune max_workers
        if not args.dse:
            specs = read_architecture_template(args.config)
            system = template_to_system(specs)
            simdram = system.device
            print(simdram.info())
            if args.use_strategy:
                print(f"Using strategy from {args.use_strategy}")
                strategy = Strategy.get_mapping_from_json(args.use_strategy)
            else:
                print("Running mapping search...")
                strategy = model.find_simdram_mapping(simdram, debug=True)
            if args.log_strategy:
                print(f"Logging strategy to {args.log_strategy}")
                strategy.to_json(args.log_strategy)
            print("Running simulation...")
            latency = model.compile_and_simulate(
                simdram, compile_mode="specific", strategy=strategy, debug=True
            )
            csv_data.append([
                f"x{simdram.compute_module.bank.device_data_width}",
                simdram.compute_module.parallelisms,
                simdram.compute_module.bank.arr_rows,
                simdram.compute_module.bank.arr_cols,
                model.stats.strategy.tile_mapping,
                model.stats.strategy.arr_mapping,
                latency,
                model.stats.tiling_utilization,
                model.stats.simd_utilization,
                model.stats.capacity_utilization,
                simdram.compute_module.capacity / 1024 / 1024 / 1024,  # in GB
            ])
            print(f"GEMM latency: {latency}ms")
        else:
            for x in ["x4", "x8", "x16"]:
                if x == "x4":
                    col_arr_count_range = col_arr_count_x4
                    device_count_range = device_count_x4
                elif x == "x8":
                    col_arr_count_range = col_arr_count_x8
                    device_count_range = device_count_x8
                elif x == "x16":
                    col_arr_count_range = col_arr_count_x16
                    device_count_range = device_count_x16

                for rank_count in rank_count_range:
                    for col_count, arr_count in col_arr_count_range:
                        for device_count in device_count_range:
                            for row_count in row_count_range:
                                futures.append(
                                    executor.submit(
                                        run_simulation,
                                        x, rank_count, col_count, arr_count,
                                        device_count, row_count
                                    )
                                )

            for future in as_completed(futures):
                try:
                    result = future.result()
                    csv_data.append(result)
                except Exception as e:
                    print(f"Simulation failed: {e}")
    if not args.dse:
        filename = f'test_gemm_{args.config.replace("configs/", "")}_{M}_{K}_{N}_simdram_ddr5_operandocality.csv'
    else:
        filename = f'test_gemm_dse_{M}_{K}_{N}_simdram_ddr5_operandocality.csv'
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(csv_header)
        writer.writerows(csv_data)
    print(f"Results written to {filename}")