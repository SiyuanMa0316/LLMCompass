from software_model.matmul import Matmul
from software_model.utils import TilingStrategy, data_type_dict, Tensor
from hardware_model.device import device_dict
from design_space_exploration.dse import template_to_system, read_architecture_template
M=1
K=12288
N=1024
model = Matmul(data_type=data_type_dict["int8"])
_ = model(
    Tensor([M, K], data_type_dict["int8"]),
    Tensor([K, N], data_type_dict["int8"]),
)

specs = read_architecture_template("configs/SIMDRAM_STD.json")
system = template_to_system(specs)

simdram = system.device
print (f"simdram config: {simdram.compute_module.bank_count}banks, {simdram.compute_module.bank.arr_count}arrays x {simdram.compute_module.bank.arr_rows}rows x {simdram.compute_module.bank.arr_cols}cols x {simdram.compute_module.bank.device_count}devices, with_PE: {simdram.compute_module.with_PE}")
print (f"simdram bw: {simdram.compute_module.bandwidth}B/s")
print (f"memory capacity: {simdram.memory_module.memory_capacity}B")
print (f"external bandwidth: {simdram.io_module.bandwidth}B/s")


# latency = model.compile_and_simulate(simdram, "heuristic-SIMDRAM")
# print(f"Siyuan's GEMM latency: {latency}s")


base = ['A', 'B', 'D']
dims = ['N', 'K']

tiling_base = ['NABDK', 'NABKD', 'NAKBD', 'NKABD', 'NABKD', 'NADKB', 'NBDKA', 'NKADB']
gemv_tiling_base = ['NDBKA']
gem_vtiling = []
for tile in gemv_tiling_base:
    res = TilingStrategy.tiling_pattern_extraction(tile)
    # print(res)
    gem_vtiling.append(res)


arr_map = [{"K": "R", "N": "C"},{"K": "C", "N": "R"}]
with_PE = [True, False]
broad_cast = ['AB' , 'A', 'B', '']
loop_order = ['mkn' , 'mnk', 'kmn', 'kmn', 'nkm', 'nmk']


tilingStrategy = []

for t in gem_vtiling:
    for a in arr_map:
        for b in broad_cast:
            for p in with_PE:
                for l in loop_order:
                    res = TilingStrategy(t, a, l, p, b)
                tilingStrategy.append(res)
                # print(res)

for i in range(len(tilingStrategy)):
    tile = tilingStrategy[i]
    print(f'*' * 20)
    print(f"Tiling Strategy[{i}]:{tile}")
    max_latency = model.compile_and_simulate(simdram,compile_mode="heuristic-SIMDRAM-Max", tilingStrategy=tile,debug=True)
    print(f"Max's GEMM latency: {max_latency*1e-6}ms")
    print(f'*' * 20)

    max_latency = 0


# A100_specs = read_architecture_template("configs/GA100x1_fp16.json")
# A100_system = template_to_system(A100_specs)
# A100_pcb = A100_system.device

# a100_latency = model.compile_and_simulate(A100_pcb, "heuristic-GPU")
# print(f"a100_latency's GEMM latency: {a100_latency}s")


# latency = model.compile_and_simulate(simdram, "heuristic-SIMDRAM-broadcast")
# print(f"Siyuan's GEMM latency: {latency}s")

# latency = model.compile_and_simulate(simdram, "heuristic-SIMDRAM-Max")
# print(f"Max's GEMM latency: {latency*1e-9}s")
