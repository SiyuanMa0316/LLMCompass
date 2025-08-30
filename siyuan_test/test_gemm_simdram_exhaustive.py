from software_model.matmul import Matmul
from software_model.utils import data_type_dict, Tensor
from software_model.mapping import Mapping
from design_space_exploration.dse import template_to_system, read_architecture_template
import csv
import argparse
parser = argparse.ArgumentParser(description="Test GEMM on DRAM PIM")
parser.add_argument("--config", type=str, help="Path to the config file")
parser.add_argument("-M", type=int, default=1024, help="Matrix A rows")
parser.add_argument("-K", type=int, default=12288, help="Matrix A columns / Matrix B rows")
parser.add_argument("-N", type=int, default=12288, help="Matrix B columns")
args = parser.parse_args()


M = args.M
K = args.K
N = args.N
model = Matmul(data_type=data_type_dict["int8"])
_ = model(
    Tensor([M, K], data_type_dict["int8"]),
    Tensor([K, N], data_type_dict["int8"]),
)


specs = read_architecture_template(args.config)
system = template_to_system(specs)

simdram = system.device
# print (f"simdram config: {simdram.compute_module.bank_count}banks, {simdram.compute_module.bank.arr_count}arrays x {simdram.compute_module.bank.arr_rows}rows x {simdram.compute_module.bank.arr_cols}cols x {simdram.compute_module.bank.device_count}devices, with_PE: {simdram.compute_module.with_PE}")
# print (f"simdram bw: {simdram.compute_module.bandwidth}B/s")
# print (f"memory capacity: {simdram.memory_module.memory_capacity}B")
# print (f"external bandwidth: {simdram.io_module.bandwidth}B/s")
print(simdram.info())

csv_data = []



# latency = model.compile_and_simulate(simdram, compile_mode="exhaustive", debug=False)
strategy = model.find_simdram_mapping(simdram, debug=False)
with open('test_gemm_simdram_dse.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        csv_header = model.stats.get_csv_header()
        csv_data = model.dse_csv_data
        csv_data.insert(0, csv_header)
        writer.writerows(csv_data)
        
        csv_data
print(f"optimal strategy: {strategy}")

# strategy.tile_mapping = Mapping.tile_mapping_extraction("MDRCNAKB")
# strategy.tile_mapping = Mapping.tile_mapping_extraction("MADNBKRC")
# strategy.arr_mapping = Mapping.arr_mapping_extraction("RKCMN")
# print(model.stats)
# print(model.stats.toCSV())
latency = model.compile_and_simulate(simdram, compile_mode="specific", strategy=strategy, debug=True)
csv_data.append(model.stats.toCSV())
csv_header = model.stats.get_csv_header()
csv_data.insert(0, csv_header)

with open('test_gemm_simdram.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(csv_data)
print(f"GEMM latency: {latency}s")


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
