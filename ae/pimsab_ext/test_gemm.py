from software_model.matmul import Matmul
from software_model.utils import data_type_dict, Tensor
from hardware_model.device import device_dict
from design_space_exploration.dse import template_to_system, read_architecture_template
M=1024
K=12288
N=49152
model = Matmul(data_type=data_type_dict["int4"])
_ = model(
    Tensor([M, K], data_type_dict["int4"]),
    Tensor([K, N], data_type_dict["int4"]),
)

specs = read_architecture_template("configs/PIMSAB.json")
system = template_to_system(specs)

pcb = system.device
latency = model.compile_and_simulate(pcb, "heuristic-PIMSAB-sim-v2")
print(latency)