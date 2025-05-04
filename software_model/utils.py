from typing import List
from utils import size


class DataType:
    def __init__(self, name: str, word_size: int) -> None:
        self.name = name
        # self.word_size:int = word_size
        #Siyuan: add support for int4 or lower
        self.word_size = word_size

rcd = 14.167
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


class Mapping:
    def __init__(self, tile_mapping: dict, arr_mapping: dict, 
                 loop_order: str='mkn', PE_enable = False, broadcast: str = "AB", weight_resident = False, input_resident=False, output_resident=False) -> None:
        self.tile_mapping = tile_mapping
        self.arr_mapping = arr_mapping
        self.loop_order = loop_order
        self.with_PE = PE_enable
        self.broadcast = broadcast
        self.weight_resident = weight_resident
        self.input_resident = input_resident
        self.output_resident = output_resident
        
    def __str__(self) -> str:
        tile_mapping_str = "".join([f"  {key}: {value}" for key, value in self.tile_mapping.items()])
        arr_mapping_str = "".join([f"  {key}: {value}" for key, value in self.arr_mapping.items()])
        return (f"Strategy(\n"
                f"  Loop Order: {self.loop_order}"
                f"  PE Enabled: {self.with_PE}"
                f"  Broadcast: {self.broadcast}\n"
                f"  Tile Mapping:{tile_mapping_str}"
                f"  Array Mapping:{arr_mapping_str}\n"
                f"  Weight Resident: {self.weight_resident}\n"
                f"  Input Resident: {self.input_resident}\n"
                f"  Output Resident: {self.output_resident}\n)")

    @staticmethod
    def tile_mapping_extraction(s):
        result = {'M': None, 'N': None, 'K': None}
        # Ensure all characters are unique and meet the required conditions
        chars = set(s)
        required = {'A', 'B', 'D'}
        assert required.issubset(chars), "A, B, D must be present"
        assert 2 <= len(chars & {'M', 'N', 'K'}) <= 3, "At least two of M, N, K must be present"
        assert len(s) == len(chars), f"{s}: All characters must be unique"

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
            result[target] = ''.join(matching) if matching else ''
        return result
    
    @staticmethod
    def arr_mapping_extraction(s):
        result = {'R': '', 'C': ''}
        # Ensure all characters are unique and meet the required conditions
        chars = set(s)
        required = {'M', 'K', 'N'}
        assert required.issubset(chars), "M, K, N must be present"
        assert 1 <= len(chars & {'R', 'C'}) <= 2, "At least two of M, N, K must be present"
        assert len(s) == len(chars), f"{s}: All characters must be unique"

        for target in ['R', 'C']:
            if target not in s:
                continue  # Keep the default value of None
            idx = s.index(target)
            matching = []
            # Search forward until a non-A/B/D character or another M/N/K is encountered
            for i in range(idx + 1, len(s)):
                char = s[i]
                if char in {'M', 'K', 'N'}:
                    matching.append(char)
                elif char in {'R', 'C'}:
                    break  # Stop when encountering another M/N/K
                else:
                    assert False, "Invalid character"  # Should never occur according to input constraints
            result[target] = ''.join(matching) if matching else ''
        return result


#find the denominator of an integer that is closest to its square root
def find_closest_divisor(n):
    assert n >= 0
    if n <= 1:
        return 1
    for i in range(int(n**0.5), 0, -1):
        if n % i == 0:
            return i

class Stats:
    def __init__ (self, strategy:Mapping) -> None:
        self.strategy = strategy
        self.tile_size = {'M': 0, 'N': 0, 'K': 0}
        self.arr_tile_size = {'M': 0, 'N': 0, 'K': 0}
        self.latency = 0
        self.compute_latency = 0
        self.io_latency = 0
        self.simd_utilization = 0
        self.capacity_utilization = 0

        #broadcast/multicast unit requirement
        self.col_multicast = False
        self.col_broadcast = False
        self.arr_broadcast = False
        self.bank_broadcast = False
        #popcount adder requirement
        self.col_popcount = False
        self.arr_popcount = False
        self.bank_popcount = False
        self.device_popcount = False

        #parse hardware requirements from tiling
        tile_mapping = self.strategy.tile_mapping
        simd_set = ['A', 'B', 'D']
        broadcast_requirement = {'A': False, 'B': False, 'D': False}
        reduction_requirement = {'A': False, 'B': False, 'D': False}
        for c in simd_set:
            # if not tiled along A/B/D
            # we need to duplicate this tile to corresponding place
            if tile_mapping['N']:
                if c not in tile_mapping['N']:
                    # MK broadcast
                    broadcast_requirement[c] = True
            if tile_mapping['K']:
                if c in tile_mapping['K']:
                    # MN reduction
                    reduction_requirement[c] = True
            if tile_mapping['M']:
                # if weight resident, no need to have hardware to broadcast weight.
                if c not in tile_mapping['M'] and not self.strategy.weight_resident:
                    # KN broadcast
                    broadcast_requirement[c] = True
        self.arr_broadcast = broadcast_requirement['A']
        self.bank_broadcast = broadcast_requirement['B']
        #device broadcast is not supported
        self.arr_popcount = reduction_requirement['A']
        self.bank_popcount = reduction_requirement['B']
        self.device_popcount = reduction_requirement['D']

        #parse hardware requirements from array mapping
        arr_mapping = self.strategy.arr_mapping
        if 'K' in arr_mapping['C']:
            self.col_popcount = True
        # if C has 2 mapped dimensions, col multicast is required. For example, in RKCMN, MK and KN matrices needs to be multicast to C, as they cannot occupy all cols in the array
        # when arr_mapping['C'] == 'MK' or arr_mapping['C'] == 'KM', only KN matrix needs to be multicasted, but if weight_resident is enabled, so KN does not need to be multicasted
        if len(arr_mapping['C']) > 1 and not ( self.strategy.weight_resident and ( arr_mapping['C'] == 'MK' or arr_mapping['C'] == 'KM')):
            self.col_multicast = True
        # if C has only 1 mapped dimension, the matrix that does not have this dimension needs to be broadcast to C
        if len(arr_mapping['C']) == 1:
            #MK broadcast to col
            if 'M' not in arr_mapping['C'] and 'K' not in arr_mapping['C']:
                self.col_broadcast = True
            #KN broadcast to col
            if 'K' not in arr_mapping['C'] and 'N' not in arr_mapping['C'] and not self.strategy.weight_resident:
                self.col_broadcast = True
            #MN does not need to be broadcast to col as it is the output matrix

    def to_string(self, debug=False, debug_info=None) -> str:
        strategy_str = str(self.strategy)

        base_output = (
            "================ Strategy Summary ================\n"
            f"{strategy_str}\n\n"
            "---------------- Tile Sizes ----------------------\n"
            "| Tile_M      | Tile_N      | Tile_K      |\n"
            "|-------------|-------------|-------------|\n"
            f"| {self.tile_size['M']:<11} | {self.tile_size['N']:<11} | {self.tile_size['K']:<11} |\n\n"
            "| Arr_Tile_M  | Arr_Tile_N  | Arr_Tile_K  |\n"
            "|-------------|-------------|-------------|\n"
            f"| {self.arr_tile_size['M']:<11} | {self.arr_tile_size['N']:<11} | {self.arr_tile_size['K']:<11} |\n\n"
            "--------------- Utilization ----------------------\n"
            f"| SIMD Utilization     | {self.simd_utilization:<22}%|\n"
            f"| Capacity Utilization | {self.capacity_utilization:<22}%|\n\n"
            "------------------ Latency -----------------------\n"
            f"| Total Latency        | {self.latency:<22} cycles|\n"
            f"| Compute Latency      | {self.compute_latency:<22} cycles|\n"
            f"| IO Latency           | {self.io_latency:<22} cycles|\n\n"
            "----------- Hardware Requirements ---------------\n"
            f"| col_multicast        | {str(self.col_multicast):<22}|\n"
            f"| col_broadcast        | {str(self.col_broadcast):<22}|\n"
            f"| arr_broadcast        | {str(self.arr_broadcast):<22}|\n"
            f"| bank_broadcast       | {str(self.bank_broadcast):<22}|\n"
            f"| col_popcount         | {str(self.col_popcount):<22}|\n"
            f"| arr_popcount         | {str(self.arr_popcount):<22}|\n"
            f"| bank_popcount        | {str(self.bank_popcount):<22}|\n"
            f"| device_popcount      | {str(self.device_popcount):<22}|\n"
            "=================================================="
        )

        if debug and debug_info:
            debug_output = (
                "\n\n---------------- Debug Info -----------------------\n"
                "| Remaining Tiles       |                         |\n"
                "|-----------------------|-------------------------|\n"
                f"| M_remain              | {debug_info.get('M_remain', 'N/A'):<23}|\n"
                f"| N_remain              | {debug_info.get('N_remain', 'N/A'):<23}|\n"
                f"| K_remain              | {debug_info.get('K_remain', 'N/A'):<23}|\n\n"
                "| Fully Partitioned #Tiles |                       |\n"
                "|-------------------------|-----------------------|\n"
                f"| M_t                   | {debug_info.get('M_t', 'N/A'):<23}|\n"
                f"| N_t                   | {debug_info.get('N_t', 'N/A'):<23}|\n"
                f"| K_t                   | {debug_info.get('K_t', 'N/A'):<23}|\n\n"
                "|------------------------|------------------------|\n"
                "=================================================="
            )
            return f"{base_output}{debug_output}"

        return base_output



    def __str__(self) -> str:
        return self.to_string(debug=False)

    def get_csv_header(self):
        return ['loop_order', 'with_PE', 'broadcast',
                'tiling_M', 'tiling_N', 'tiling_K',
                'arr_map_R', 'arr_map_C',
                'tile_M', 'tile_N', 'tile_K',
                'arr_tile_M', 'arr_tile_N', 'arr_tile_K',
                'SIMD_Utilization', 'Capacity_Utilization',
                'latency', 'compute_latency', 'io_latency',
                'col_multicast', 'col_broadcast',
                'arr_broadcast', 'bank_broadcast',
                'col_popcount', 'arr_popcount',
                'bank_popcount', 'device_popcount']
    def toCSV(self):
        return [self.strategy.loop_order,
                self.strategy.with_PE,
                self.strategy.broadcast,
                self.strategy.tile_mapping['M'],
                self.strategy.tile_mapping['N'],
                self.strategy.tile_mapping['K'],
                self.strategy.arr_mapping['R'],
                self.strategy.arr_mapping['C'],
                self.tile_size['M'],
                self.tile_size['N'],
                self.tile_size['K'],
                self.arr_tile_size['M'],
                self.arr_tile_size['N'],
                self.arr_tile_size['K'],
                self.simd_utilization,
                self.capacity_utilization,
                self.latency,
                self.compute_latency,
                self.io_latency,
                self.col_multicast,
                self.col_broadcast,
                self.arr_broadcast,
                self.bank_broadcast,
                self.col_popcount,
                self.arr_popcount,
                self.bank_popcount,
                self.device_popcount]
if __name__ == '__main__':
    # Demo testcases
    s = "MKNABD"
    res = Mapping.tile_mapping_extraction(s)
    print(res)
    s= "RMNCK"
    res = Mapping.arr_mapping_extraction(s)
    print(res)
    # assert res == expect, "Extracted result does not match"