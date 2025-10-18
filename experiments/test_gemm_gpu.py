from software_model.matmul import Matmul
from software_model.utils import data_type_dict, Tensor
from hardware_model.device import device_dict
from design_space_exploration.dse import template_to_system, read_architecture_template
import argparse
parser = argparse.ArgumentParser(description="Test GEMM on GPU")
parser.add_argument("--config", type=str, default="configs/H100x1_sxm5_int8.json", help="Path to the config file")
parser.add_argument("--M", type=int, default=1, help="Matrix M dimension")
parser.add_argument("--K", type=int, default=12288*2, help="Matrix K dimension")
parser.add_argument("--N", type=int, default=12288*2, help="Matrix N dimension")
parser.add_argument("--precision", type=str, default="int8", help="Data precision")
args = parser.parse_args()

M = args.M
K = args.K
N = args.N
precision = args.precision
print(f"Simulate GEMM {M}x{K} * {K}x{N} with {precision} on GPU")
model = Matmul(data_type=data_type_dict[precision]) 
_ = model(
    Tensor([M, K], data_type_dict[precision]),
    Tensor([K, N], data_type_dict[precision]),
)

specs = read_architecture_template(args.config)
system = template_to_system(specs)

# pcb.compute_module.l2_size = system.device.compute_module.l2_size / 2
# pcb.compute_module.core.SRAM_size = system.device.compute_module.core.SRAM_size
# pcb.io_module.bandwidth = pcb.io_module.bandwidth / 2
# print(f"Core SRAM size: {system.device.compute_module.core.SRAM_size/1024}KB")
# print(f"L2 size: {system.device.compute_module.l2_size/1000/1000}MB")
# print(f"HBM bandwidth: {system.device.io_module.bandwidth/1000/1000/1000}GB/s")
print(system.device.info())
latency = model.compile_and_simulate(system.device, "heuristic-GPU")
latency_roofline = model.roofline_model(system.device)
print(f"Simulated latency: {latency}")
print(f"Roofline latency: {latency_roofline}")