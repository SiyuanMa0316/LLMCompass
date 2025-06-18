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
        self.row_buffer_size = arr_cols * arr_count * device_count
        self.device_data_width = device_data_width
        self.total_data_width = device_data_width * device_count
        self.effective_freq = effective_freq
        self.bandwidth = self.effective_freq * self.total_data_width / 8



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
    "SIMDRAM": Overhead(2.1e-5, 1.2e-5, 4.5e-5, 4.5e-5),
}



class ComputeModuleSIMDRAM:
    def __init__(
        self,
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
        self.bandwidth = bank.bandwidth * bank_count * rank_count * channel_count
        self.with_PE = with_PE
        self.bit_parallel = bit_parallel
        # if with_PE:
        #     if bit_parallel:
        #         self.op_latency_dict = simdram_PE_op_latency_dict
        #     else:
        #         self.op_latency_dict = simdram_op_latency_dict
        if bit_parallel:
            self.gops = channel_count * rank_count * bank_count * bank.arr_count * bank.subarr_count * bank.arr_cols / 8 *2 / (14.16*6)
        else:
            self.gops = channel_count * rank_count * bank_count * bank.arr_count * bank.subarr_count * bank.arr_cols *2 / 453.3
        self.parallelisms = {}
        self.parallelisms['C'] = self.channel_count
        self.parallelisms['R'] = self.rank_count
        self.parallelisms['B'] = self.bank_count
        self.parallelisms['A'] = self.bank.arr_count
        self.parallelisms['S'] = self.bank.subarr_count
        self.parallelisms['D'] = self.bank.device_count
        

    def info(self):
        info_str = (f"simdram config: {self.channel_count}channels x {self.rank_count}ranks x {self.bank_count}banks x {self.bank.arr_count}arrays x {self.bank.subarr_count}subarrays x {self.bank.subarr_rows}subarr_rows x {self.bank.arr_cols}cols x {self.bank.device_count}devices, with_PE: {self.with_PE}\n"
                    f"simdram bw: {self.bandwidth/1e9}GB/s\n"
                    f"simdram ops: {self.gops/1e3}Tops\n"
                    )
        return info_str


compute_module_simdram_dict = {
    "simdram_standard": ComputeModuleSIMDRAM(
        1,
        1,
        bank_dict["Bank_ddr4_2400"],
        16,
        overhead_dict["SIMDRAM"],
    ),
    "simdram_96x": ComputeModuleSIMDRAM(
        12,
        8,
        bank_dict["Bank_ddr4_2400"],
        16,
        True,
        overhead_dict["SIMDRAM"],
    ),
}
