from design_space_exploration.dse import template_to_system, read_architecture_template
from software_model.utils import Tensor, DataType, data_type_dict
from software_model.transformer import TransformerBlockAutoRegressionTP, TransformerBlockInitComputationTP
from hardware_model.device import device_dict
from software_model.matmul import Matmul
from scalesim.scale_sim import scalesim
import os
import pandas as pd


simdram_spec = read_architecture_template("configs/SIMDRAM_STD.json")
simdram_sys = template_to_system(simdram_spec)

#setup tensor size
M = 1024
K = 12288
N = 1024

# print(f"Testing Matmul Workload with size {M} x {K} x {N}")
# print(f"Total GEMM Storage Size: Input + Output {(M * K + K * N + M * N) * 16/1024/1024/1024/8}GB")
# print(f"System Specs: IO-Bandiwdth {simdram_sys.device.io_module.bandwidth/1024/1024/1024} GBps  Memory Capacity {simdram_sys.device.memory_module.memory_capacity/1024/1024/1024} GB")

# test_overhead = True
matmul = Matmul(data_type=data_type_dict['fp16'])
_ = matmul(Tensor([M, K]), Tensor([K, N]))
print(simdram_sys.device.type)
simulate_latency = matmul.compile_and_simulate(simdram_sys.device, compile_mode="heuristic-SIMDRAM-v2")
# print(f"SIMDRAM heuristic Matmul Latency {simulate_latency} ns")




