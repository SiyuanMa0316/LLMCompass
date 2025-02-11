from cProfile import label
from design_space_exploration.dse import template_to_system, read_architecture_template
from software_model.utils import Tensor, DataType, data_type_dict, simdram_op_latency_dict
from software_model.transformer import TransformerBlockAutoRegressionTP, TransformerBlockInitComputationTP
from hardware_model.device import device_dict
from software_model.matmul import Matmul
from scalesim.scale_sim import scalesim
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

simdram_spec = read_architecture_template("configs/SIMDRAM_STD.json")
simdram_sys = template_to_system(simdram_spec)

#setup tensor size
M = 1
K = 12288
pow = 20

seq = [f"{i}" for i in range(pow)]
# print(seq)
N = 1024
gemv_latency = []
gemm_latency = []
standard_latency = []

T = simdram_op_latency_dict["fp16"]["add"] + simdram_op_latency_dict['fp16']["mul"]

a = 64
b = 16
d = 8
col = 128
frac = T / (col * a * b * d)


matmul = Matmul(data_type_dict['fp16'])


_ = matmul(Tensor([M, K]), Tensor([K, N]))
simulate_latency = matmul.compile_and_simulate(simdram_sys.device, compile_mode="heuristic-SIMDRAM-v2", debug = True)

# test_overhead = True
# matmul = Matmul(data_type=data_type_dict['fp16'])
# for i in range(pow):
#     N = 2 ** i
#     _ = matmul(Tensor([M, K]), Tensor([K, N]))
#     simulate_latency = matmul.compile_and_simulate(simdram_sys.device, compile_mode="heuristic-SIMDRAM-v2")
#     standard_latency.append((M * N * K * frac)/ 1000000)
#     gemv_latency.append(simulate_latency / 1000000)


# N = 1024
# for i in range(pow):
#     M = 2 ** i
#     _ = matmul(Tensor([M, K]), Tensor([K, N])) 
#     simulate_latency = matmul.compile_and_simulate(simdram_sys.device, compile_mode="heuristic-SIMDRAM-v2")
#     standard_latency.append((M * N * K * frac)/ 1000000)
#     gemm_latency.append(simulate_latency / 1000000)



# # Plotting the results
# plt.figure(figsize=(12, 6))

# # Plot 1: gemv_latency
# plt.subplot(1, 2, 1)
# plt.plot(seq, gemv_latency, marker='o', label='gemv_latency')
# plt.plot(seq, standard_latency[0:pow], marker='*', label='standard_latency')
# plt.xlabel('log2(N)')
# plt.ylabel('latency ms')
# plt.title('N vs gemv_latency')

# # Plot 2: gemm_latency
# plt.subplot(1, 2, 2)

# plt.plot(seq, gemm_latency, marker='o', label='gemm_latency')
# plt.plot(seq, standard_latency[pow:], marker='*', label='standard_latency')
# plt.xlabel('log2(M)')
# plt.ylabel('latency ms')
# plt.title('M vs gemm_latency')

# plt.tight_layout()
# plt.legend()
# plt.show()



