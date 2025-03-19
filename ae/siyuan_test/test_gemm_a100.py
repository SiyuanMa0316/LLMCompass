from software_model.matmul import Matmul
from software_model.utils import data_type_dict, Tensor
from hardware_model.device import device_dict
from design_space_exploration.dse import template_to_system, read_architecture_template

M=1024
K=12288
N=1024
precision = "int8"
model = Matmul(data_type=data_type_dict[precision])
_ = model(
    Tensor([M, K], data_type_dict[precision]),
    Tensor([K, N], data_type_dict[precision]),
)

specs = read_architecture_template(f"configs/GA100x1_{precision}.json")
system = template_to_system(specs)

pcb = system.device
# pcb.compute_module.l2_size = system.device.compute_module.l2_size / 2
# pcb.compute_module.core.SRAM_size = system.device.compute_module.core.SRAM_size
# pcb.io_module.bandwidth = pcb.io_module.bandwidth / 2
print(f"Core SRAM size: {system.device.compute_module.core.SRAM_size/1024}KB")
print(f"L2 size: {system.device.compute_module.l2_size/1024/1024}MB")
print(f"HBM bandwidth: {pcb.io_module.bandwidth/1024/1024/1024}GB/s")
latency = model.compile_and_simulate(pcb, "heuristic-GPU")
print(latency)