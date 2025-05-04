from software_model.matmul import Matmul
from software_model.utils import Mapping, data_type_dict, Tensor
from design_space_exploration.dse import template_to_system, read_architecture_template
import csv
from software_model.relayout import Relayout
M_1=1024
K_1=12288
N_1=12288
matmul_1 = Matmul(data_type=data_type_dict["int8"])
_ = matmul_1(
    Tensor([M_1, K_1], data_type_dict["int8"]),
    Tensor([K_1, N_1], data_type_dict["int8"]),
)

M_2=M_1
K_2=N_1
N_2=12288
matmul_2 = Matmul(data_type=data_type_dict["int8"])
_ = matmul_2(
    Tensor([M_2, K_2], data_type_dict["int8"]),
    Tensor([K_2, N_2], data_type_dict["int8"]),
)

specs = read_architecture_template("configs/SIMDRAM_STD.json")
system = template_to_system(specs)

simdram = system.device
print (f"simdram config: {simdram.compute_module.bank_count}banks, {simdram.compute_module.bank.arr_count}arrays x {simdram.compute_module.bank.arr_rows}rows x {simdram.compute_module.bank.arr_cols}cols x {simdram.compute_module.bank.device_count}devices, with_PE: {simdram.compute_module.with_PE}")
print (f"simdram bw: {simdram.compute_module.bandwidth}B/s")
print (f"memory capacity: {simdram.memory_module.memory_capacity}B")
print (f"external bandwidth: {simdram.io_module.bandwidth}B/s")


# csv_data = []

# tile_mapping_list = ['MANBKD', 'MABNKD', 'MNABKD', 'MDNKAB']
# arr_mapping_list = ['RKNCM','RMKCN','RMNCK','RNCMK', 'RMCKN', 'RKCMN']
# arr_map_list = ['RKNCM']

tile_mapping = Mapping.tile_mapping_extraction('MANBKD')
arr_mapping = Mapping.arr_mapping_extraction('RKNCM') 
strategy = Mapping(tile_mapping, arr_mapping, loop_order='mkn', PE_enable=True, broadcast = 'AB', weight_resident=True)


tile_mapping_2 = Mapping.tile_mapping_extraction('MANDKB')
arr_mapping_2 = Mapping.arr_mapping_extraction('RKNCM')
strategy_2 = Mapping(tile_mapping_2, arr_mapping_2, loop_order='mkn', PE_enable=True, broadcast = 'AB', weight_resident=True)

relayout = Relayout(simdram, matmul_1, strategy, matmul_2, strategy_2)
# inplace_relayout_latency = 0
# if not relayout.need_host():
#     strategy.output_resident = True
#     strategy_2.input_resident = True
#     inplace_relayout_latency = relayout.get_inplace_relayout_latency()
# else:
#     strategy.output_resident = False
#     strategy_2.input_resident = False
#     inplace_relayout_latency = 0
inplace_relayout_latency = relayout.inplace_relayout_latency

latency_1 = matmul_1.compile_and_simulate(simdram, compile_mode="specific", strategy=strategy, debug=False)
        # print(model.stats)
        # print(model.stats.toCSV())

latency_2 = matmul_2.compile_and_simulate(simdram, compile_mode="specific", strategy=strategy_2, debug=False)
total_latency = latency_1 + inplace_relayout_latency + latency_2
print(f"GEMM_1 latency: {latency_1}ms, relayout latency: {inplace_relayout_latency}ms, GEMM_2 latency: {latency_2}ms, total latency: {total_latency}ms")
# A100_specs = read_architecture_template("configs/GA100x1_int8.json")
# A100_system = template_to_system(A100_specs)
# A100_pcb = A100_system.device

# a100_latency = model.compile_and_simulate(A100_pcb, "heuristic-GPU")
# print(f"a100_latency's GEMM latency: {a100_latency}s")


# latency = model.compile_and_simulate(simdram, "heuristic-SIMDRAM-broadcast")
# print(f"Siyuan's GEMM latency: {latency}s")

# latency = model.compile_and_simulate(simdram, "heuristic-SIMDRAM-Max")
# print(f"Max's GEMM latency: {latency*1e-9}s")
