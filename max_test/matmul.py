from design_space_exploration.dse import template_to_system, read_architecture_template
from software_model.utils import Tensor, DataType, data_type_dict
from software_model.transformer import TransformerBlockAutoRegressionTP, TransformerBlockInitComputationTP
from hardware_model.device import device_dict
from software_model.matmul import Matmul
from scalesim.scale_sim import scalesim
import os
import pandas as pd

pimsab_spec = read_architecture_template("configs/PIMSAB.json")
pimsab_sys = template_to_system(pimsab_spec)

#setup tensor size
M = 1024
K = 12288
N = K
# A100_device = device_dict['A100_80GB_fp16']

print(f"Testing Matmul Workload with size {M} x {K} x {N}")

# test_overhead = True
matmul = Matmul(data_type=data_type_dict['fp16'])
_ = matmul(Tensor([M, K]), Tensor([K, N]))
"""
Starting the scalsim simulation based on PIMSAB configuration
"""




# matmul.gpu_kernel_launch_overhead()
# test_overhead = False
# gpu_real_latency = matmul.run_on_gpu()


# roofline_latency = matmul.roofline_model(A100_device)
# simulate_latency = matmul.compile_and_simulate(A100_device, compile_mode="exhaustive")
# print(f"A100_device {simulate_latency}")


#simulate systolic array specs of PIMSAB
pimsab_sa_height = pimsab_sys.device.compute_module.core.systolic_array.array_height
pimsab_sa_width = pimsab_sys.device.compute_module.core.systolic_array.array_width
pimsab_mac_perclock = pimsab_sys.device.compute_module.core.systolic_array.mac_per_cycle
config_dir = "./systolic_array_model/temp/"
config = os.path.join(config_dir, f"systolic_array_{os.getpid()}.cfg")
os.makedirs(config_dir, exist_ok=True)
print(f"Generating config file {config}")
data_flow = 'os'
with open(config, "w") as f:
    f.writelines("[general]\n")
    f.writelines("run_name = systolic_array\n\n")
    f.writelines("[architecture_presets]\n")
    f.writelines("ArrayHeight:    " + str(pimsab_sa_height) + "\n")
    f.writelines("ArrayWidth:     " + str(pimsab_sa_width) + "\n")
    f.writelines("IfmapSramSzkB:    " + str(1024) + "\n")
    f.writelines("FilterSramSzkB:   " + str(1024) + "\n")
    f.writelines("OfmapSramSzkB:    " + str(1024) + "\n")
    f.writelines("IfmapOffset:    0\n")
    f.writelines("FilterOffset:   10000000\n")
    f.writelines("OfmapOffset:    20000000\n")
    f.writelines("Dataflow : " + data_flow + "\n")
    f.writelines("Bandwidth : " + "100" + "\n")
    f.writelines("MemoryBanks: 1\n\n")
    f.writelines("[run_presets]\n")
    f.writelines("InterfaceBandwidth: CALC\n")
topology = f"./systolic_array_model/temp/matmul_{os.getpid()}.csv"
with open(topology, "w") as f:
    f.writelines("Layer, M, N, K\n")
    f.writelines(f"matmul1, {M}, {N}, {K},\n")
print(f"Starting Scalsim Simulation: Generating topology {topology}")
logpath = f"./systolic_array_model/temp/"
s = scalesim(
    save_disk_space=True,
    verbose=False,
    config=config,
    topology=topology,
    input_type_gemm=True,
)
print(f"starting simulation")
s.run_scale(top_path=logpath)
print(f"simulation finished")
cycle_count = s.runner.single_layer_sim_object_list[0].total_cycles
util_rate = s.runner.single_layer_sim_object_list[0].overall_util
print(f"Writing to csv files")
with open(
    f"./systolic_array_model/look_up_table_{pimsab_sa_height}_{pimsab_sa_width}.csv",
    "a",
) as f:
    f.writelines(
        f"{M},{N},{K},{pimsab_sa_height},{pimsab_sa_width},{data_flow},{cycle_count},{util_rate:.3f}\n"
    )
look_up_table = pd.DataFrame()
look_up_table.loc[(M, N, K, pimsab_sa_height, pimsab_sa_width, data_flow), :] = [
    cycle_count,
    util_rate,
]
if len(look_up_table) % 10 == 0:
    look_up_table.sort_index(inplace=True)
# look_up_table.to_csv(f"./systolic_array_model/look_up_table_{pimsab_sa_height}_{pimsab_sa_width}.csv")
print(f"Appended to file: look_up_table_{pimsab_sa_height}_{pimsab_sa_width}.csv")


# cycles = matmul.simulate_systolic_array_cycle_count(look_up_table, M, K, N, pimsab_sa_height, pimsab_sa_width, pimsab_mac_perclock)
# print(f"PIMSAB SA cycles {cycles}")

# pimsab_latency = matmul.compile_and_simulate(pimsab_sys.device, compile_mode="exhaustive")


# ops = 2 * M * N * K / 1e12
# # real_tflops = ops / gpu_real_latency

# pimsab_simulate_tflops = ops / pimsab_latency

# print(f"Simulate PIMSAB TFlops {pimsab_simulate_tflops}\n")



