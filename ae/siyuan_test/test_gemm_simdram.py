from software_model.matmul import Matmul
from software_model.utils import data_type_dict, Tensor
from hardware_model.device import device_dict
from design_space_exploration.dse import template_to_system, read_architecture_template
M=1024
K=12288
N=12288
model = Matmul(data_type=data_type_dict["int8"])
_ = model(
    Tensor([M, K], data_type_dict["int8"]),
    Tensor([K, N], data_type_dict["int8"]),
)

specs = read_architecture_template("configs/SIMDRAM_STD.json")
system = template_to_system(specs)

simdram = system.device
print (f"simdram config: {simdram.compute_module.bank_count}banks, {simdram.compute_module.bank.arr_count}arrays x {simdram.compute_module.bank.arr_rows}rows x {simdram.compute_module.bank.arr_cols}cols x {simdram.compute_module.bank.device_count}devices")
print (f"simdram external bw: {simdram.compute_module.bandwidth}B/s")
print (f"memory capacity: {simdram.memory_module.memory_capacity}B")
print (f"memory bandwidth: {simdram.io_module.bandwidth}B/s")


latency = model.compile_and_simulate(simdram, "heuristic-SIMDRAM")
print(f"Siyuan's GEMM latency: {latency}s")

latency = model.compile_and_simulate(simdram, "heuristic-SIMDRAM-Max")
print(f"Max's GEMM latency: {latency*1e-9}s")