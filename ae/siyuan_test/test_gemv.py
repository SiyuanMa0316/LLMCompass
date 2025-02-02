from software_model.matmul import Matmul
from software_model.utils import data_type_dict, Tensor
from hardware_model.device import device_dict
from design_space_exploration.dse import template_to_system_pimsab, read_architecture_template
M=1
K=25600
N=25600
model = Matmul(data_type=data_type_dict["int8"])
_ = model(
    Tensor([M, K], data_type_dict["int8"]),
    Tensor([K, N], data_type_dict["int8"]),
)

specs = read_architecture_template("configs/PIMSAB.json")
system = template_to_system_pimsab(specs)
print(f"pimsab config: {system.device.compute_module.tile_count}tiles, {system.device.compute_module.tile.arr_count}arrays x {system.device.compute_module.tile.arr_rows}rows x {system.device.compute_module.tile.arr_cols}cols")
print(f"pimsab NoC bw: {system.device.compute_module.noc.bandwidth}B/s")
print(f"pimsab dram total bw: {system.device.io_module.bandwidth}B/s")
print(f"pimsab dram bw per tile in pipeline mode: {system.device.io_module.bandwidth/system.device.compute_module.tile_count}B/s")

pcb = system.device
latency = model.compile_and_simulate(pcb, "heuristic-PIMSAB-sim-v2")
print(latency)