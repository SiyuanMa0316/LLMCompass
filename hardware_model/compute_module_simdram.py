from math import ceil
from software_model.utils import DataType, data_type_dict


class Bank:
    def __init__(
        self,
        device_count,
        arr_count,
        subarr_count,
        arr_cols,
        arr_rows,
        device_data_width,
        effective_freq
        
    ):
        self.device_count = device_count
        self.arr_count = arr_count
        self.subarr_count = subarr_count
        self.arr_cols = arr_cols
        self.arr_rows = arr_rows
        self.subarr_rows = arr_rows // subarr_count
        # self.row_buffer_size = arr_cols * arr_count * device_count
        self.device_data_width = device_data_width
        self.total_data_width = device_data_width * device_count
        self.effective_freq = effective_freq
        self.bandwidth = self.effective_freq * self.total_data_width / 8
        self.internal_bandwidth = self.device_count * self.arr_count * self.arr_cols / 14.16e-9 /8 # assuming 14.16ns for one array activate
        self.capacity = (
            self.device_count * self.arr_count * self.arr_cols * self.arr_rows / 8
        )  # in bytes

        #logical view of architecture
        global_bitline_bus = True
        if global_bitline_bus:
            self.global_bitline_width = 256
            self.logical_PEs = self.global_bitline_width * 8 #int8 computation takes 9 cycles for each bit, each cycle takes 1 ns. We set PEs can be feed within 8ns to hide that latency
            self.logical_arr_cols = self.global_bitline_width
            self.logical_arr_count = self.logical_PEs // self.logical_arr_cols #always 8
            self.logical_arr_rows = self.arr_rows * self.arr_cols * self.arr_count // self.logical_arr_cols // self.logical_arr_count
            self.logical_subarr_rows = self.logical_arr_rows // subarr_count
        else:
            self.logical_PEs = self.arr_count * self.arr_cols
            self.logical_arr_cols = self.arr_cols
            self.logical_arr_count = self.arr_count
            self.logical_arr_rows = self.arr_rows
            self.logical_subarr_rows = self.subarr_rows




bank_dict = {
    "Bank_ddr4_2400": Bank(8, 64, 32, 128, 131072, 8, 2400e6)
}




class Overhead:
    def __init__(self, matmul, softmax, layernorm, gelu):
        self.matmul = matmul
        self.softmax = softmax
        self.layernorm = layernorm
        self.gelu = gelu


overhead_dict = {
    "A100": Overhead(2.1e-5, 1.2e-5, 4.5e-5, 4.5e-5),
    "TPUv3": Overhead(11e-5, 30e-5, 14e-5, 10e-5),
    "MI210": Overhead(3.4e-5, 2.2e-5, 2.8e-5, 2.1e-5),
    "PIMSAB": Overhead(2.1e-5, 1.2e-5, 4.5e-5, 4.5e-5),
    "SIMDRAM": Overhead(0, 0, 0, 0),
}



class ComputeModuleSIMDRAM:
    def __init__(
        self,
        host_channel_count: int,
        channel_count,
        rank_count,
        bank: Bank,
        bank_count,
        with_PE: bool,
        bit_parallel: bool = False,
        overhead: Overhead = overhead_dict["SIMDRAM"],
    ):
        self.channel_count = channel_count
        self.rank_count = rank_count
        self.bank = bank
        self.bank_count = bank_count
        self.clock_freq = bank.effective_freq
        self.overhead = overhead
        self.bandwidth = bank.bandwidth * bank_count * rank_count * host_channel_count
        self.internal_bandwidth = bank.internal_bandwidth * bank_count * rank_count * host_channel_count
        self.with_PE = with_PE
        self.bit_parallel = bit_parallel
        # if with_PE:
        #     if bit_parallel:
        #         self.op_latency_dict = simdram_PE_op_latency_dict
        #     else:
        #         self.op_latency_dict = simdram_op_latency_dict
        if bit_parallel:
            self.gops = channel_count * rank_count * bank_count * bank.logical_arr_count * bank.subarr_count * bank.logical_arr_cols / 8 *2 / (14.16*6)
        else:
            self.gops = channel_count * rank_count * bank_count * bank.logical_arr_count * bank.subarr_count * bank.device_count * bank.logical_arr_cols *2 / (16*12.2)
        self.parallelisms = {}
        self.parallelisms['C'] = self.channel_count
        self.parallelisms['R'] = self.rank_count
        self.parallelisms['B'] = self.bank_count
        self.parallelisms['A'] = self.bank.logical_arr_count
        self.parallelisms['S'] = self.bank.subarr_count
        self.parallelisms['D'] = self.bank.device_count
        self.capacity = self.bank.capacity * self.bank_count * self.rank_count * self.channel_count

        #area model
        broadcaster_area = {'64b_1_1': 0, '64b_1_2': 6683.3/4, '64b_1_4': 6683.3/2, '64b_1_8': 6683.3, '64b_1_16': 13217, '64b_1_32': 26845.36, '64b_1_64': 53535.39} # in um^2
        bank_broadcast_total_area = broadcaster_area[f'64b_1_{self.bank_count}'] * self.rank_count * self.channel_count /1E6   # in mm^2
        rank_broadcast_total_area = broadcaster_area[f'64b_1_{self.rank_count}'] * self.channel_count /1E6  # in mm^2
        array_broadcast_total_area = broadcaster_area[f'64b_1_{self.bank.logical_arr_count}'] * self.bank.device_count * self.bank_count * self.rank_count * self.channel_count /1E6  # in mm^2
        # popcount_unit_area = 671.6 # in um^2
        popcount_unit_area = 13812 # in um^2

        popcount_total_area = popcount_unit_area * self.bank.logical_arr_count * self.bank.device_count * self.bank_count * self.rank_count * self.channel_count /1E6  # in mm^2
        # pe_area = 18.3 #in um^2
        pe_area = 67 #in um^2
        pe_total_area = pe_area * self.bank.logical_PEs* self.bank.device_count * self.bank_count * self.rank_count * self.channel_count /1E6  # in mm^2
        sram_per_bit_area = 0.296 # in um^2
        sram_total_area = sram_per_bit_area * 17 * self.bank.logical_PEs * self.bank.device_count * self.bank_count * self.rank_count * self.channel_count /1E6  # in mm^2
        # print(f"simdram area breakdown: bank_broadcast:{bank_broadcast_total_area}mm^2, rank_broadcast:{rank_broadcast_total_area}mm^2, array_broadcast:{array_broadcast_total_area}mm^2, popcount:{popcount_total_area}mm^2, pe:{pe_total_area}mm^2, sram:{sram_total_area}mm^2")
        pnr_factor = 1.64
        tech_node_factor = (14/45) ** 2 # assuming area scales with square of tech node
        self.peripheral_area = (bank_broadcast_total_area + rank_broadcast_total_area + array_broadcast_total_area + popcount_total_area + pe_total_area + sram_total_area)* pnr_factor * tech_node_factor # in mm^2
        self.dram_area = self.capacity *8 / 1024 / 1024 / 1024 /16 * 66 # in mm^2, assume 16Gb chip and 66mm^2 per chip

    def info(self):
        info_str = (f"simdram config: {self.channel_count}channels x {self.rank_count}ranks x {self.bank_count}banks x {self.bank.arr_count}arrays x {self.bank.subarr_count}subarrays x {self.bank.subarr_rows}subarr_rows x {self.bank.arr_cols}cols x {self.bank.device_count}devices, with_PE: {self.with_PE}\n"
                    f"simdram logical config: {self.channel_count}channels x {self.rank_count}ranks x {self.bank_count}banks x {self.bank.logical_arr_count}arrays x {self.bank.subarr_count}subarrays x {self.bank.logical_subarr_rows}subarr_rows x {self.bank.logical_arr_cols}cols x {self.bank.device_count}devices, with_PE: {self.with_PE}\n"
                    f"simdram bw: {self.bandwidth/1e9}GB/s\n"
                    f"simdram internal bw: {self.internal_bandwidth/1e9}GB/s\n"
                    f"simdram ops: {self.gops/1e3}Tops\n"
                    f"memory capacity: {self.capacity / 1024 / 1024 / 1024}GB\n"
                    f"peripheral area: {self.peripheral_area}mm^2\n"
                    f"dram area: {self.dram_area}mm^2\n"
                    )
        return info_str


# compute_module_simdram_dict = {
#     "simdram_standard": ComputeModuleSIMDRAM(
#         1,
#         1,
#         1,
#         bank_dict["Bank_ddr4_2400"],
#         16,
#         overhead_dict["SIMDRAM"],
#     ),
#     "simdram_96x": ComputeModuleSIMDRAM(
#         12,
#         12,
#         8,
#         bank_dict["Bank_ddr4_2400"],
#         16,
#         True,
#         overhead_dict["SIMDRAM"],
#     ),
# }
