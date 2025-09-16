from software_model.matmul import Matmul
from software_model.utils import data_type_dict, Tensor
from software_model.mapping import Mapping
from design_space_exploration.dse import template_to_system, read_architecture_template
import csv
M=1024
K=12288
N=12288
model = Matmul(data_type=data_type_dict["int8"])
_ = model(
    Tensor([M, K], data_type_dict["int8"]),
    Tensor([K, N], data_type_dict["int8"]),
)

specs = read_architecture_template("configs/ddr5.json")
system = template_to_system(specs)

row_count_range = [16*2**10, 32*2**10, 64*2**10, 128*2**10] # 16K, 32K, 64K, 128K
col_arr_count_x8 = [(256,64), (512,32), (1024,16)]
col_arr_count_x4 = [(256,32), (512,16), (1024,8)]
col_arr_count_x16 = [(256,64), (512,32), (1024,16)]
bank_count_x8 = 32
bank_count_x4 = 32
bank_count_x16 = 16
device_count_x8 = [8, 16]
device_count_x4 = [16, 32]
device_count_x16 = [4, 8]
rank_count_range = [1, 2, 4, 8, 16, 32, 64]
channel_count = 10

csv_header = ["parallelism","col", "latency", "tiling_utilization", "col_utilization", "capacity_utilization", "total_capacity"]
csv_data = []
for x in ["x4", "x8", "x16"]:
    if x == "x4":
        col_arr_count_range = col_arr_count_x4
        bank_count = bank_count_x4
        device_count_range = device_count_x4
    elif x == "x8":
        col_arr_count_range = col_arr_count_x8
        bank_count = bank_count_x8
        device_count_range = device_count_x8
    elif x == "x16":
        col_arr_count_range = col_arr_count_x16
        bank_count = bank_count_x16
        device_count_range = device_count_x16
    for rank_count in rank_count_range:
        for col_count, arr_count in col_arr_count_range:
            for device_count in device_count_range:
                for row_count in row_count_range:
                    specs["compute_module"]["rank_count"] = rank_count
                    specs["compute_module"]["bank"]["array_cols"] = col_count
                    specs["compute_module"]["bank"]["array_rows"] = row_count
                    specs["compute_module"]["bank_count"] = bank_count
                    specs["compute_module"]["bank"]["array_count"] = arr_count
                    specs["compute_module"]["bank"]["device_count"] = device_count
                    specs["compute_module"]["channel_count"] = channel_count

                    system = template_to_system(specs)
                    simdram = system.device
                    print(simdram.info())

                    strategy = model.find_simdram_mapping(simdram, debug=False)

                    latency = model.compile_and_simulate(simdram, compile_mode="specific", strategy=strategy, debug=True)
                    csv_data.append([simdram.compute_module.parallelisms, simdram.compute_module.bank.arr_cols, latency, model.stats.tiling_utilization, model.stats.simd_utilization, model.stats.capacity_utilization, simdram.compute_module.capacity / 1024 / 1024 / 1024])  # capacity in GB
        

with open('test_gemm_simdram_dse_ddr5_jeeho_operandocality.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(csv_header)
        writer.writerows(csv_data)

