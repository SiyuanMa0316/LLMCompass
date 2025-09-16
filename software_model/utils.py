from typing import List
from utils import size


class DataType:
    def __init__(self, name: str, word_size: float) -> None:
        self.name = name
        # self.word_size:int = word_size
        #Siyuan: add support for int4 or lower
        self.word_size = word_size #bytes
        self.bits = word_size * 8 #bits

# rcd = 14.167
rcd = 12.2 #ddr5 6400  https://de.wikipedia.org/wiki/DDR-SDRAM
faw=13.312#https://www.igorslab.de/en/intel-vs-jedec-ddr5-timings-in-extreme-technical-practice-test/
data_type_dict = {"int4": DataType("int4", 0.5), "int8": DataType("int8", 1), "fp16": DataType("fp16", 2), "fp32": DataType("fp32", 4)}
simdram_op_latency_dict = {
    DataType("int8", 1).name: {"add": 3121, "mul": 31815, 'add_reduce':3121+8*rcd, 'mul_reduce':31815+16*rcd},
    DataType("int16", 1).name: {"add": 6193, "mul": 131135, 'add_reduce':6193+16*rcd, 'mul_reduce':131135+32*rcd},
    DataType("int32", 1).name: {"add": 12337, "mul": 532143, 'add_reduce':12337+32*rcd, 'mul_reduce':532143+64*rcd}
}
simdram_PE_op_latency_dict = {
    DataType("int8", 1).name: {"add": 340, "mul": 453.3, 'add_reduce':340, 'mul_reduce':453.3},
    DataType("int16", 1).name: {"add": 680, "mul": 906, 'add_reduce':680, 'mul_reduce': 906},
    DataType("int32", 1).name: {"add": 1360, "mul": 1813, 'add_reduce':1360, 'mul_reduce': 1813}
}

class Tensor:
    def __init__(
        self, shape: List, data_type=data_type_dict["fp16"]
    ) -> None:
        self.shape = shape
        self.size = size(shape)
        self.data_type = data_type





#find the denominator of an integer that is closest to its square root
def find_closest_divisor(n):
    assert n >= 0
    if n <= 1:
        return 1
    for i in range(int(n**0.5), 0, -1):
        if n % i == 0:
            return i


