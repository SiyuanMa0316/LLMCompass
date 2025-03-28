from typing import List
from utils import size


class DataType:
    def __init__(self, name: str, word_size: int) -> None:
        self.name = name
        # self.word_size:int = word_size
        #Siyuan: add support for int4 or lower
        self.word_size = word_size

data_type_dict = {"int4": DataType("int4", 0.5), "int8": DataType("int8", 1), "fp16": DataType("fp16", 2), "fp32": DataType("fp32", 4)}
simdram_op_latency_dict = {
    DataType("int8", 1).name: {"add": 3121, "mul": 31815, 'add_reduce':3121+340},
    DataType("int16", 1).name: {"add": 6193, "mul": 131135, 'add_reduce':6193+680},
    DataType("int32", 1).name: {"add": 12337, "mul": 532143, 'add_reduce':12337+1360}
}
simdram_PE_op_latency_dict = {
    DataType("int8", 1).name: {"add": 340, "mul": 453.3, 'add_reduce':340},
    DataType("int16", 1).name: {"add": 680, "mul": 906, 'add_reduce':680},
    DataType("int32", 1).name: {"add": 1360, "mul": 1813, 'add_reduce':1360}
}

class Tensor:
    def __init__(
        self, shape: List, data_type=data_type_dict["fp16"]
    ) -> None:
        self.shape = shape
        self.size = size(shape)
        self.data_type = data_type


class TilingStrategy:
    def __init__(self, tiling: dict, arr_mapping: dict, 
                 loop_order: str='mkn', PE_enable = False, broadcast: str = "AB") -> None:
        self.tiling = tiling
        self.arr_mapping = arr_mapping
        self.loop_order = loop_order
        self.with_PE = PE_enable
        self.broadcast = broadcast

    def __str__(self) -> str:
        tiling_str = "".join([f"  {key}: {value}" for key, value in self.tiling.items()])
        arr_mapping_str = "".join([f"  {key}: {value}" for key, value in self.arr_mapping.items()])
        return (f"TilingStrategy(\n"
                f"  Loop Order: {self.loop_order}"
                f"  PE Enabled: {self.with_PE}"
                f"  Broadcast: {self.broadcast}\n"
                f"  Tiling Parameters:{tiling_str}"
                f"  Array Mapping:{arr_mapping_str}\n)")

    @staticmethod
    def tiling_pattern_extraction(s):
        result = {'M': None, 'N': None, 'K': None}
        # Ensure all characters are unique and meet the required conditions
        chars = set(s)
        required = {'A', 'B', 'D'}
        assert required.issubset(chars), "A, B, D must be present"
        assert 2 <= len(chars & {'M', 'N', 'K'}) <= 3, "At least two of M, N, K must be present"
        assert len(s) == len(chars), "All characters must be unique"

        for target in ['M', 'N', 'K']:
            if target not in s:
                continue  # Keep the default value of None
            idx = s.index(target)
            matching = []
            # Search forward until a non-A/B/D character or another M/N/K is encountered
            for i in range(idx + 1, len(s)):
                char = s[i]
                if char in {'A', 'B', 'D'}:
                    matching.append(char)
                elif char in {'M', 'N', 'K'}:
                    break  # Stop when encountering another M/N/K
                else:
                    assert False, "Invalid character"  # Should never occur according to input constraints
            result[target] = ''.join(matching) if matching else None
        return result




if __name__ == '__main__':
    s = "MKNABD"
    res = TilingStrategy.tiling_pattern_extraction(s)
    print(res)
    # assert res == expect, "Extracted result does not match"