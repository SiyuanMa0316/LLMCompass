from software_model.matmul import Matmul
from software_model.utils import TilingStrategy, data_type_dict, Tensor
from hardware_model.device import device_dict
from design_space_exploration.dse import template_to_system, read_architecture_template
import csv
M=1
K=12288
N=1024
model = Matmul(data_type=data_type_dict["int8"])
_ = model(
    Tensor([M, K], data_type_dict["int8"]),
    Tensor([K, N], data_type_dict["int8"]),
)

specs = read_architecture_template("configs/SIMDRAM_STD.json")
system = template_to_system(specs)

simdram = system.device
print (f"simdram config: {simdram.compute_module.bank_count}banks, {simdram.compute_module.bank.arr_count}arrays x {simdram.compute_module.bank.arr_rows}rows x {simdram.compute_module.bank.arr_cols}cols x {simdram.compute_module.bank.device_count}devices, with_PE: {simdram.compute_module.with_PE}")
print (f"simdram bw: {simdram.compute_module.bandwidth}B/s")
print (f"memory capacity: {simdram.memory_module.memory_capacity}B")
print (f"external bandwidth: {simdram.io_module.bandwidth}B/s")

csv_header = model.stats.get_csv_header()
csv_data = [csv_header]

tiling_list = ['MNABKD', 'MNAKBD', 'MNKABD']
arr_map_list = ['RMKCN','RMNCK', 'RMCKN']
# arr_map_list = ['RKNCM']

for tiling_str in tiling_list:
    for arr_map_str in arr_map_list:
        tiling = TilingStrategy.tiling_pattern_extraction(tiling_str)
        arr_map = TilingStrategy.mapping_extraction(arr_map_str)
        with_PE = True
        broadcast = 'AB'
        loop_order = 'mkn' 


        strategy = TilingStrategy(tiling, arr_map, loop_order, with_PE, broadcast)

        latency = model.compile_and_simulate(simdram,compile_mode="heuristic-SIMDRAM-broadcast", strategy=strategy, debug=True)
        # print(model.stats)
        # print(model.stats.toCSV())
        csv_data.append(model.stats.toCSV())

with open('test_gemv_simdram.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(csv_data)
# print(f"GEMM latency: {latency}ms")


# A100_specs = read_architecture_template("configs/GA100x1_int8.json")
# A100_system = template_to_system(A100_specs)
# A100_pcb = A100_system.device

# a100_latency = model.compile_and_simulate(A100_pcb, "heuristic-GPU")
# print(f"a100_latency's GEMM latency: {a100_latency}s")


# latency = model.compile_and_simulate(simdram, "heuristic-SIMDRAM-broadcast")
# print(f"Siyuan's GEMM latency: {latency}s")

# latency = model.compile_and_simulate(simdram, "heuristic-SIMDRAM-Max")
# print(f"Max's GEMM latency: {latency*1e-9}s")
