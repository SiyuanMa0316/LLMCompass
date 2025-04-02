from software_model.matmul import Matmul
from software_model.utils import data_type_dict, Tensor
from hardware_model.device import device_dict
from design_space_exploration.dse import template_to_system, read_architecture_template
M=1024
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


# latency = model.compile_and_simulate(simdram, "heuristic-SIMDRAM")
# print(f"Siyuan's GEMM latency: {latency}s")
# max_latency = model.compile_and_simulate(simdram, "heuristic-SIMDRAM-Max")
# print(f"Max's GEMM latency: {max_latency*1e-9}s")

# A100_specs = read_architecture_template("configs/GA100x1_fp16.json")
# A100_system = template_to_system(A100_specs)
# A100_pcb = A100_system.device

# a100_latency = model.compile_and_simulate(A100_pcb, "heuristic-GPU")
# print(f"a100_latency's GEMM latency: {a100_latency}s")


latency = model.compile_and_simulate(simdram, "heuristic-SIMDRAM-broadcast")
print(f"Siyuan's GEMM latency: {latency}s")

# latency = model.compile_and_simulate(simdram, "heuristic-SIMDRAM-Max")
# print(f"Max's GEMM latency: {latency*1e-9}s")
