from software_model.matmul import Matmul
from software_model.utils import data_type_dict, Tensor
from hardware_model.device import device_dict
from design_space_exploration.dse import template_to_system_pimsab, read_architecture_template
M=256
K=12288
N=12288
model = Matmul(data_type=data_type_dict["int8"])
_ = model(
    Tensor([M, K], data_type_dict["int8"]),
    Tensor([K, N], data_type_dict["int8"]),
)

specs = read_architecture_template("configs/PIMSAB.json")
system = template_to_system_pimsab(specs)

pcb = system.device
# pcb.compute_module.tile_count = 12
# pcb.io_module.bandwidth = pcb.io_module.bandwidth / 2
latency = model.compile_and_simulate(pcb, "heuristic-PIMSAB-sim-v3")
print(latency)