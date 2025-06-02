from software_model.matmul import Matmul
from software_model.utils import  data_type_dict, Tensor
from software_model.mapping import Mapping
from design_space_exploration.dse import template_to_system, read_architecture_template
import csv
M=1
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
print(simdram.info())


csv_data = []



# latency = model.compile_and_simulate(simdram, compile_mode="exhaustive", debug=False)
strategy = model.find_simdram_mapping(simdram, debug=False)
print(f"optimal strategy: {strategy}")
# print(model.stats)
# print(model.stats.toCSV())
latency = model.compile_and_simulate(simdram, compile_mode="specific", strategy=strategy, debug=True)
csv_data.append(model.stats.toCSV())
csv_header = model.stats.get_csv_header()
csv_data.insert(0, csv_header)

with open('test_gemv_simdram.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(csv_data)
print(f"GEMM latency: {latency}ms")


# latency = model.compile_and_simulate(simdram, compile_mode="exhaustive", debug=False)
# print(f"test_gemm_exhastive: GEMM latency: {latency}ms")
# A100_specs = read_architecture_template("configs/GA100x1_int8.json")
# A100_system = template_to_system(A100_specs)
# A100_pcb = A100_system.device

# a100_latency = model.compile_and_simulate(A100_pcb, "heuristic-GPU")
# print(f"a100_latency's GEMM latency: {a100_latency}s")


# latency = model.compile_and_simulate(simdram, "heuristic-SIMDRAM-broadcast")
# print(f"Siyuan's GEMM latency: {latency}s")

# latency = model.compile_and_simulate(simdram, "heuristic-SIMDRAM-Max")
# print(f"Max's GEMM latency: {latency*1e-9}s")
