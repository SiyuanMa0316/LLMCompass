from software_model.mapping import Mapping
from hardware_model.device import Device
class TileStats:
    def __init__(self, tile_size=None, arr_tile_size=None, M_K_io_latency=None, K_N_io_latency=None, M_N_io_latency=None, tile_compute_latency=None, arr_latency=None, K_reduction_latency=None, tiling_utilization=None, col_utilization=None, capacity_utilization=None) -> None:
        self.tile_size = tile_size
        self.arr_tile_size = arr_tile_size
        self.M_N_io_latency = M_N_io_latency
        self.M_K_io_latency = M_K_io_latency
        self.K_N_io_latency = K_N_io_latency
        
        self.arr_latency = arr_latency
        self.K_reduction_latency = K_reduction_latency
        if M_K_io_latency is not None and K_N_io_latency is not None and M_N_io_latency is not None and tile_compute_latency is not None and K_reduction_latency is not None:
            self.latency = M_K_io_latency + K_N_io_latency + M_N_io_latency + tile_compute_latency + K_reduction_latency
        self.compute_latency = tile_compute_latency
        self.array_latency = arr_latency
        self.reduction_latency = K_reduction_latency
        if M_K_io_latency is not None and K_N_io_latency is not None and M_N_io_latency is not None:
            self.io_latency = M_K_io_latency + K_N_io_latency + M_N_io_latency
        self.col_utilization = col_utilization
        self.tiling_utilization = tiling_utilization
        self.capacity_utilization = capacity_utilization

    def __str__(self):
        return (f"Tile Size: {self.tile_size}, Arr Tile Size: {self.arr_tile_size}, "
                f"M_K IO Latency: {self.M_K_io_latency}, K_N IO Latency: {self.K_N_io_latency}, "
                f"M_N IO Latency: {self.M_N_io_latency}, Tile Compute Latency: {self.compute_latency}, "
                f"Array Latency: {self.array_latency}, K Reduction Latency: {self.reduction_latency}, "
                f"Total Latency: {self.latency}, Col Utilization: {self.col_utilization}, "
                f"Tiling Utilization: {self.tiling_utilization}, Capacity Utilization: {self.capacity_utilization}")
    def get_csv_header(self):
        return ['tile_size', 'arr_tile_size', 'M_K_io_latency', 'K_N_io_latency', 'M_N_io_latency',
                'compute_latency', 'array_latency', 'reduction_latency', 'latency',
                'col_utilization', 'tiling_utilization', 'capacity_utilization']
    def toCSV(self):
        return [self.tile_size, self.arr_tile_size, self.M_K_io_latency, self.K_N_io_latency, self.M_N_io_latency,
                self.compute_latency, self.array_latency, self.reduction_latency, self.latency,
                self.col_utilization, self.tiling_utilization, self.capacity_utilization]
class Stats:
    def __init__ (self, device:Device,  strategy:Mapping) -> None:
        self.strategy = strategy
        self.tile_size = {'M': 0, 'N': 0, 'K': 0}
        self.arr_tile_size = {'M': 0, 'N': 0, 'K': 0}
        self.latency = 0
        self.compute_latency = 0
        self.total_array_latency = 0
        self.total_reduction_latency = 0
        self.io_latency = 0
        self.simd_utilization = 0
        self.total_simd_lane = 0
        self.used_simd_lane = 0
        # self.tiling_utilization = {"A": 0, "B": 0, "D": 0, "R": 0, "C": 0}
        self.tiling_utilization = {c: 0 for c in device.compute_module.parallelisms.keys()}
        self.capacity_utilization = 0

        # #broadcast/multicast unit requirement
        self.col_multicast = False
        self.col_broadcast = False
        # self.arr_broadcast = False
        # self.bank_broadcast = False
        # self.rank_broadcast = False
        # self.channel_broadcast = False
        # #popcount adder requirement
        self.col_popcount = False
        # self.arr_popcount = False
        # self.bank_popcount = False
        # self.device_popcount = False

        #parse hardware requirements from tiling
        tile_mapping = self.strategy.tile_mapping
        simd_set = device.compute_module.parallelisms.keys()
        # broadcast_requirement = {'A': False, 'B': False, 'D': False, 'R': False, 'C': False}
        self.broadcast_requirement = {c: False for c in simd_set}
        # reduction_requirement = {'A': False, 'B': False, 'D': False, 'R': False, 'C': False}
        self.reduction_requirement = {c: False for c in simd_set}
        for c in simd_set:
            # if not tiled along A/B/D
            # we need to duplicate this tile to corresponding place
            if tile_mapping['N']:
                if c in tile_mapping['N'] and not self.strategy.input_resident:
                    # MK broadcast
                    self.broadcast_requirement[c] = True
            if tile_mapping['K']:
                if c in tile_mapping['K']:
                    # MN reduction
                    self.reduction_requirement[c] = True
            if tile_mapping['M']:
                # if weight resident, no need to have hardware to broadcast weight.
                if c in tile_mapping['M'] and not self.strategy.weight_resident:
                    # KN broadcast
                    self.broadcast_requirement[c] = True
        # self.arr_broadcast = broadcast_requirement['A']
        # self.bank_broadcast = broadcast_requirement['B']
        # self.rank_broadcast = broadcast_requirement['R']
        # self.channel_broadcast = broadcast_requirement['C']
        # #device broadcast is not supported
        # self.rank_broadcast = broadcast_requirement['R']
        # self.channel_broadcast = broadcast_requirement['C']
        # self.arr_popcount = reduction_requirement['A']
        # self.bank_popcount = reduction_requirement['B']
        # self.device_popcount = reduction_requirement['D']
        # self.rank_popcount = reduction_requirement['R']
        # self.channel_popcount = reduction_requirement['C']
        

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
            "| Row         | Col         | Array       | Bank        | Device      | Rank        | Channel     |\n"
            "|-------------|-------------|-------------|-------------|-------------|-------------|-------------|\n"
            f"| {self.capacity_utilization:<11} | {self.simd_utilization:<11} | {self.tiling_utilization['A']:<11} | {self.tiling_utilization['B']:<11} | {self.tiling_utilization['D']:<11} | {self.tiling_utilization['R']:<11} | {self.tiling_utilization['C']:<11} |\n\n"
            "------------------ Latency -----------------------\n"
            f"| Total Latency        | {self.latency:<22} cycles|\n"
            f"| Total Compute Latency | {self.compute_latency:<22} cycles|\n"
            f"| Total Array Latency  | {self.total_array_latency:<22} cycles|\n"
            f"| Total Reduction Latency| {self.total_reduction_latency:<22} cycles|\n"
            f"| IO Latency           | {self.io_latency:<22} cycles|\n\n"
            "----------- Hardware Requirements ---------------\n"
            f"| col_multicast        | {str(self.col_multicast):<22}|\n"
            f"| col_broadcast        | {str(self.col_broadcast):<22}|\n"
            f"| broadcast            | {str(self.broadcast_requirement):<22}|\n"
            # f"| bank_broadcast       | {str(self.bank_broadcast):<22}|\n"
            # f"| rank_broadcast       | {str(self.rank_broadcast):<22}|\n"
            # f"| channel_broadcast    | {str(self.channel_broadcast):<22}|\n"
            f"| reduction            | {str(self.reduction_requirement):<22}|\n"
            # f"| col_popcount         | {str(self.col_popcount):<22}|\n"
            # f"| arr_popcount         | {str(self.arr_popcount):<22}|\n"
            # f"| bank_popcount        | {str(self.bank_popcount):<22}|\n"
            # f"| device_popcount      | {str(self.device_popcount):<22}|\n"
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
                'row_Utilization', 'col_Utilization', 'array_Utilization', 'subarray_Utilization',
                'bank_Utilization', 'device_Utilization', 'rank_Utilization', 'channel_Utilization',
                'latency', 'compute_latency', 'array_latency', 'reduction_latency', 'io_latency',
                'col_multicast', 'col_broadcast',
                'subarr_broadcast', 'arr_broadcast', 'bank_broadcast', 'rank_broadcast', 'channel_broadcast',
                'col_popcount', 'arr_popcount',
                'bank_popcount', 'device_popcount', 'rank_popcount', 'channel_popcount',]
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
                self.capacity_utilization,
                self.simd_utilization,
                self.tiling_utilization['A'],
                self.tiling_utilization['S'],
                self.tiling_utilization['B'],
                self.tiling_utilization['D'],
                self.tiling_utilization['R'],
                self.tiling_utilization['C'],
                self.latency,
                self.compute_latency,
                self.total_array_latency,
                self.total_reduction_latency,
                self.io_latency,
                self.col_multicast,
                self.col_broadcast,
                self.broadcast_requirement['S'],
                self.broadcast_requirement['A'],
                self.broadcast_requirement['B'],
                self.broadcast_requirement['R'],
                self.broadcast_requirement['C'],
                self.col_popcount,
                self.reduction_requirement['A'],
                self.reduction_requirement['B'],
                self.reduction_requirement['D'],
                self.reduction_requirement['R'],
                self.reduction_requirement['C'],
               ]