from math import ceil
from software_model.utils import DataType, data_type_dict


class Bank:
    def __init__(
        self,
        device_count,
        arr_count,
        arr_cols,
        arr_rows,
        device_data_width,
        effective_freq
        
    ):
        self.device_count = device_count
        self.arr_count = arr_count
        self.arr_cols = arr_cols
        self.arr_rows = arr_rows
        self.row_buffer_size = arr_cols * arr_count * device_count
        self.device_data_width = device_data_width
        self.total_data_width = device_data_width * device_count
        self.effective_freq = effective_freq
        self.bandwidth = self.effective_freq * self.total_data_width / 8

        


bank_dict = {
    "Bank_ddr4_2400": Bank(8, 64, 128, 131072, 8, 2400e6)
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
        bank: Bank,
        bank_count,
        with_PE: bool,
        overhead: Overhead = overhead_dict["SIMDRAM"],
    ):
        self.bank = bank
        self.bank_count = bank_count
        self.clock_freq = bank.effective_freq
        self.overhead = overhead
        self.bandwidth = bank.bandwidth * bank_count
        self.with_PE = with_PE


compute_module_simdram_dict = {
    "simdram_standard": ComputeModuleSIMDRAM(
        bank_dict["Bank_ddr4_2400"],
        16,
        overhead_dict["SIMDRAM"],
    ),
}
