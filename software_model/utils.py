from typing import List
from utils import size


class DataType:
    def __init__(self, name: str, word_size: int) -> None:
        self.name = name
        self.word_size:int = word_size

data_type_dict = {"int8": DataType("int8", 1), "fp16": DataType("fp16", 2), "fp32": DataType("fp32", 4)}
simdram_op_latency_dict = {
    {DataType("int8", 1).name: {"add": 3121, "mul": 31815, "mac":44152}},
    {DataType("fp16", 1).name: {"add": 6193, "mul": 131135, "mac": 44152}},
    {DataType("fp32", 1).name: {"add": 12337, "mul": 532143, "mac": 44152}}
}

class Tensor:
    def __init__(
        self, shape: List, data_type=data_type_dict["fp16"]
    ) -> None:
        self.shape = shape
        self.size = size(shape)
        self.data_type = data_type
        
