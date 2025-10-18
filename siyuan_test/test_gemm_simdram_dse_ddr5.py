from software_model.matmul import Matmul
from software_model.utils import data_type_dict, Tensor
from software_model.strategy import Strategy
from design_space_exploration.dse import template_to_system, read_architecture_template
import csv
M=1024*16
K=12288*16
N=12288*16
model = Matmul(data_type=data_type_dict["int8"])
_ = model(
    Tensor([M, K], data_type_dict["int8"]),
    Tensor([K, N], data_type_dict["int8"]),
)

specs = read_architecture_template("configs/SIMDRAM_160x_more_bank.json")
system = template_to_system(specs)

rank_count_range = [1, 2, 4, 8, 16, 32, 64]

csv_header = ["parallelism","col", "latency", "tiling_utilization", "col_utilization", "capacity_utilization", "total_capacity"]
csv_data = []
for rank_count in rank_count_range:

    specs["compute_module"]["rank_count"] = rank_count
    system = template_to_system(specs)
    simdram = system.device
    print(simdram.info())
    strategy = model.find_simdram_mapping(simdram, debug=False)

    latency = model.compile_and_simulate(simdram, compile_mode="specific", strategy=strategy, debug=True)
    csv_data.append([simdram.compute_module.parallelisms, simdram.compute_module.bank.arr_cols, latency, model.stats.tiling_utilization, model.stats.simd_utilization, model.stats.capacity_utilization, simdram.compute_module.capacity / 1024 / 1024 / 1024])  # capacity in GB
    

with open('test_gemm_simdram_dse_ddr5.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(csv_header)
        writer.writerows(csv_data)

