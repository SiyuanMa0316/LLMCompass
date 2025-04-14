from hardware_model.device import Device
from math import ceil
from software_model.utils import TilingStrategy, simdram_op_latency_dict, simdram_PE_op_latency_dict
from software_model.utils import TilingStrategy, find_closest_divisor, Stats
import numpy as np

def get_tile_io_latency(pcb_module: Device, broad_cast: str, tile_1: int, tile_2: int, word_size: float, dup: str) -> float:
    array_per_device_bank = pcb_module.compute_module.bank.arr_count
    device = pcb_module.compute_module.bank.device_count
    bank = pcb_module.compute_module.bank_count

    array_broadcast_enable = False
    bank_broadcast_enable = False
    latency = 0
    if 'A' in broad_cast:
        array_broadcast_enable = True
    if 'B' in broad_cast:
        bank_broadcast_enable = True
    
    latency = tile_1 * tile_2 * word_size  / pcb_module.compute_module.bandwidth
    # print(f"latency {latency} = {tile_1} * {tile_2} * {word_size} / {pcb_module.compute_module.bandwidth}")
    if 'A' in dup and not array_broadcast_enable:
        # duplicate tile to every device
        latency *= array_per_device_bank
    if 'B' in dup and not bank_broadcast_enable:
        # duplicate tile to every bank
        latency *= bank
    if 'D' in dup:
        # duplicate tile to every device
        latency *= device
    return latency
   

def find_arr_tile_max(self, pcb_module: Device, strategy:TilingStrategy, debug=False):
    num_array = pcb_module.compute_module.bank.arr_count
    num_bank = pcb_module.compute_module.bank_count
    num_device = pcb_module.compute_module.bank.device_count
    tiling = strategy.tiling
    arr_tile_M_max = self.M
    arr_tile_K_max = self.K
    arr_tile_N_max = self.N
    simd_set = {'A','B','D'}
    simd_size = {'A': num_array, 'B': num_bank, 'D': num_device}
    for simd_dim in simd_set:
        # only tile when M/K/N is larger than simd_sizes.
        if simd_dim in tiling['M'] and arr_tile_M_max >= simd_size[simd_dim]:
            arr_tile_M_max = arr_tile_M_max // simd_size[simd_dim] 
        if simd_dim in tiling['K'] and arr_tile_K_max >= simd_size[simd_dim]:
            arr_tile_K_max = arr_tile_K_max // simd_size[simd_dim]
        if simd_dim in tiling['N'] and arr_tile_N_max >= simd_size[simd_dim]:
            arr_tile_N_max = arr_tile_N_max // simd_size[simd_dim]

    arr_tile_max = {'M': arr_tile_M_max, 'K': arr_tile_K_max, 'N': arr_tile_N_max}
    if debug:
        print(f'Maximum Tile Size across Bank/Device/Array: {arr_tile_max}')
    return arr_tile_max

def find_arr_tile(self, pcb_module: Device, strategy:TilingStrategy, debug=False):
    arr_tile_max = find_arr_tile_max(self, pcb_module, strategy, debug)
    col_per_arr = pcb_module.compute_module.bank.arr_cols
    row_per_arr = pcb_module.compute_module.bank.arr_rows
    arr_mapping = strategy.arr_mapping
    row_elem_per_arr = row_per_arr // (self.data_type.word_size * 8)
    if arr_mapping['C'] == 'M':
        arr_tile_M = col_per_arr
        arr_tile_N = min(arr_tile_max['N'], find_closest_divisor(row_elem_per_arr))
        arr_tile_K = row_elem_per_arr // arr_tile_N
        

    elif arr_mapping['C'] == 'N':
        arr_tile_N = col_per_arr
        arr_tile_M = min(arr_tile_max['M'], find_closest_divisor(row_elem_per_arr))
        arr_tile_K = row_elem_per_arr // arr_tile_M


    elif arr_mapping['C'] == 'K':
        arr_tile_K = col_per_arr
        arr_tile_M = min(arr_tile_max['M'], find_closest_divisor(row_elem_per_arr)) 
        arr_tile_N = row_elem_per_arr // arr_tile_M



    elif arr_mapping['C'] == 'MN' or arr_mapping['C'] == 'NM':
        arr_tile_M = min(arr_tile_max['M'], find_closest_divisor(col_per_arr))
        arr_tile_N = col_per_arr // arr_tile_M
        arr_tile_K = row_elem_per_arr // 2

    elif arr_mapping['C'] == 'MK' or arr_mapping['C'] == 'KM':
        arr_tile_M = min(arr_tile_max['M'], find_closest_divisor(col_per_arr))
        arr_tile_K = col_per_arr // arr_tile_M
        arr_tile_N = row_elem_per_arr // 2

        
    elif arr_mapping['C'] == 'NK' or arr_mapping['C'] == 'KN':
        arr_tile_N = min(arr_tile_max['N'], find_closest_divisor(col_per_arr))
        arr_tile_K = col_per_arr // arr_tile_N
        arr_tile_M = row_elem_per_arr // 2


    # if debug:

    return arr_tile_M, arr_tile_N, arr_tile_K

def get_arr_tile_latency(self, pcb_module: Device, arr_tile_M, arr_tile_N, arr_tile_K, arr_mapping, debug=False):
    col_per_arr = pcb_module.compute_module.bank.arr_cols
    row_per_arr = pcb_module.compute_module.bank.arr_rows
    ################ Compute Latencies #################
    #add and mul latency
    if pcb_module.compute_module.with_PE:
        add_op_latency = simdram_PE_op_latency_dict[self.data_type.name]['add']
        mul_op_latency = simdram_PE_op_latency_dict[self.data_type.name]['mul']
        acc_op_latency = simdram_PE_op_latency_dict['int32']['add']
        add_reduce_op_latency = simdram_PE_op_latency_dict[self.data_type.name]['add_reduce']
        mul_reduce_op_latency = simdram_PE_op_latency_dict[self.data_type.name]['mul_reduce']
    else:
        add_op_latency = simdram_op_latency_dict[self.data_type.name]['add']
        mul_op_latency = simdram_op_latency_dict[self.data_type.name]['mul']
        acc_op_latency = simdram_op_latency_dict['int32']['add']
        add_reduce_op_latency = simdram_op_latency_dict[self.data_type.name]['add_reduce']
        mul_reduce_op_latency = simdram_op_latency_dict[self.data_type.name]['mul_reduce']
    
    if arr_mapping['C'] == 'M':
        macs = arr_tile_N * arr_tile_K
        mac_latency = macs * (add_op_latency + mul_op_latency) * 1e-9
        arr_latency = mac_latency + acc_op_latency * 1e-9
        simd_utilization = arr_tile_M / col_per_arr
        capacity_utilization = (self.data_type.word_size * 8) * arr_tile_N * arr_tile_K / row_per_arr
    elif arr_mapping['C'] == 'N':
        macs = arr_tile_M * arr_tile_K
        mac_latency = macs * (add_op_latency + mul_op_latency) * 1e-9
        arr_latency = mac_latency + acc_op_latency * 1e-9
        simd_utilization = arr_tile_N / col_per_arr
        capacity_utilization = (self.data_type.word_size * 8) * arr_tile_M * arr_tile_K / row_per_arr
    elif arr_mapping['C'] == 'K':
        mul_reduce_latency = arr_tile_M * arr_tile_N * mul_reduce_op_latency * 1e-9
        arr_latency = mul_reduce_latency + acc_op_latency*1e-9
        simd_utilization = arr_tile_K / col_per_arr
        capacity_utilization = (self.data_type.word_size * 8) * arr_tile_M * arr_tile_N / row_per_arr
    elif arr_mapping['C'] == 'MN' or arr_mapping['C'] == 'NM':
        macs = arr_tile_K
        mac_latency = macs * (add_op_latency + mul_op_latency) * 1e-9
        arr_latency = mac_latency + acc_op_latency * 1e-9
        simd_utilization = arr_tile_M * arr_tile_N / col_per_arr
        capacity_utilization = (self.data_type.word_size * 8) * 2 * arr_tile_K / row_per_arr
    elif arr_mapping['C'] == 'MK' or arr_mapping['C'] == 'KM':
        mul_reduce_latency = arr_tile_N * mul_reduce_op_latency * 1e-9
        arr_latency = mul_reduce_latency + acc_op_latency*1e-9
        simd_utilization = arr_tile_M * arr_tile_K / col_per_arr
        capacity_utilization = (self.data_type.word_size * 8) * 2 * arr_tile_N / row_per_arr
    elif arr_mapping['C'] == 'NK' or arr_mapping['C'] == 'KN':
        mul_reduce_latency = arr_tile_M * mul_reduce_op_latency * 1e-9
        arr_latency = mul_reduce_latency + acc_op_latency*1e-9
        simd_utilization = arr_tile_N * arr_tile_K / col_per_arr
        capacity_utilization = (self.data_type.word_size * 8) * 2 * arr_tile_M / row_per_arr
        # if debug:
        #     print(f"get_arr_tile_latency: arr_tile_M={arr_tile_M}, arr_tile_N={arr_tile_N}, arr_tile_K={arr_tile_K}, arr_mapping={arr_mapping}, latency={arr_latency}=({arr_tile_M} * {mul_reduce_op_latency} + {acc_op_latency}) * 1e-9, parallelism_utilization={arr_tile_K/col_per_arr}, capacity_utilization={arr_tile_M*arr_tile_N/row_per_arr}")
    self.stats.simd_utilization = max(self.stats.simd_utilization, simd_utilization)
    self.stats.capacity_utilization = max(self.stats.capacity_utilization, capacity_utilization)
    # if debug:
    #     print(f"simd utilization: {simd_utilization}, capacity utilization: {capacity_utilization}")
    return arr_latency


def find_tile_size(self, pcb_module: Device, tiling, arr_tile_M, arr_tile_N, arr_tile_K, debug=False):
    num_array = pcb_module.compute_module.bank.arr_count
    num_bank = pcb_module.compute_module.bank_count
    num_device = pcb_module.compute_module.bank.device_count
    num_rank = 1
    tile_size = {'M': arr_tile_M, 'K': arr_tile_K, 'N': arr_tile_N}
    for key in tiling.keys():
        val = tiling[key]
        if val:
            for c in val:
                if c == 'A':
                    tile_size[key] *= num_array
                if c == 'B':
                    tile_size[key] *= num_bank
                if c == 'D':
                    tile_size[key] *= num_device
    tile_size['M'] = min(tile_size['M'], self.M)
    tile_size['N'] = min(tile_size['N'], self.N)
    tile_size['K'] = min(tile_size['K'], self.K)
    # if debug:
    #     print(f"find_tile_size: {tile_size}")
    return tile_size

def get_tile_latency(self, pcb_module: Device, strategy:TilingStrategy, tile_size, debug=False):
    num_array = pcb_module.compute_module.bank.arr_count
    num_bank = pcb_module.compute_module.bank_count
    num_device = pcb_module.compute_module.bank.device_count
    num_rank = 1
    tiling = strategy.tiling
    arr_tile_size = {'M': tile_size['M'], 'K': tile_size['K'], 'N': tile_size['N']}
    for key in tiling.keys():
        val = tiling[key]
        if val:
            for c in val:
                if c == 'A':
                    arr_tile_size[key] = arr_tile_size[key] // num_array
                if c == 'B':
                    arr_tile_size[key] = arr_tile_size[key] // num_bank
                if c == 'D':
                    arr_tile_size[key] = arr_tile_size[key] // num_device
    self.stats.tile_size['M'] = max(self.stats.tile_size['M'], tile_size['M'])
    self.stats.tile_size['K'] = max(self.stats.tile_size['K'], tile_size['K'])
    self.stats.tile_size['N'] = max(self.stats.tile_size['N'], tile_size['N'])
    self.stats.arr_tile_size['M'] = max(self.stats.arr_tile_size['M'], arr_tile_size['M'])
    self.stats.arr_tile_size['K'] = max(self.stats.arr_tile_size['K'], arr_tile_size['K'])
    self.stats.arr_tile_size['N'] = max(self.stats.arr_tile_size['N'], arr_tile_size['N'])
    #extract duplication of matrices
    parallelisms = ['A', 'B', 'D']
    K_N_dup = []
    M_K_dup = []
    for c in parallelisms:
        # if not tiled along A/B/D
        # we need to duplicate this tile to corresponding place
        if c not in tiling['K'] and c not in tiling['N']:
            K_N_dup.append(c)
        if c not in tiling['M'] and c not in tiling['K']:
            M_K_dup.append(c)

    #compute io latencies
    M_K_io_latency = get_tile_io_latency(pcb_module, strategy.broadcast, tile_size['M'], tile_size['K'], self.data_type.word_size, M_K_dup)
    if strategy.weight_resident:
        # if weight resident, we don't need to load K_N tile
        K_N_io_latency = 0
    else:
        K_N_io_latency = get_tile_io_latency(pcb_module, strategy.broadcast, tile_size['K'], tile_size['N'], self.data_type.word_size, K_N_dup)
    # no duplication required for M_N tile, simply write it back to Host
    M_N_io_latency = tile_size['M'] * tile_size['N'] * self.data_type.word_size / pcb_module.io_module.bandwidth
    
    # compute arr_tile size from tile size
    arr_tile_size = {'M': tile_size['M'], 'K': tile_size['K'], 'N': tile_size['N']}
    for key in tiling.keys():
        val = tiling[key]
        if val:
            for c in val:
                if c == 'A':
                    arr_tile_size[key] = arr_tile_size[key] // num_array
                if c == 'B':
                    arr_tile_size[key] = arr_tile_size[key] // num_bank
                if c == 'D':
                    arr_tile_size[key] = arr_tile_size[key] // num_device
    arr_mapping = strategy.arr_mapping
    arr_latency = get_arr_tile_latency(self, pcb_module, arr_tile_size['M'], arr_tile_size['N'], arr_tile_size['K'], arr_mapping, debug)

    K_reduction_latency =  tile_size['M'] * tile_size['N'] * self.data_type.word_size / pcb_module.compute_module.bandwidth #each device contains M_tike*N_tile partial sum data, load all these to host and reduce
    for c in parallelisms:
        if c in tiling['K']:
            if c == 'A':
                K_reduction_latency *= num_array
            if c == 'B':
                K_reduction_latency *= num_bank
            if c == 'D':
                K_reduction_latency *= num_device
    tile_compute_latency =  arr_latency + K_reduction_latency

    # if debug:
    #     print(f"get_tile_latency: tile_size: {tile_size}, arr_tile_size: {arr_tile_size}, M_K_io_latency: {M_K_io_latency}, K_N_io_latency: {K_N_io_latency}, M_N_io_latency: {M_N_io_latency}, tile_compute_latency:{tile_compute_latency} = {arr_latency}(arr_latency) + {K_reduction_latency}(K_reduction_latency)")
    return (M_K_io_latency, K_N_io_latency, M_N_io_latency, tile_compute_latency)


def heuristic_simdram_broadcast (self, pcb_module: Device, strategy: TilingStrategy, debug=False):
    '''
    M->array
    N->bank
    K->device
    arr_tile_M * arr_tile_N = arr_cols
    arr_tile_K fit into arr_rows
    loop order: NKM
    '''
    if debug:
        # print(f"Matmul: {self.input1_shape}, {self.input2_shape}, {self.output_shape}")
        print(f"--- Input Matmul Size --- M:{self.M}, K:{self.K}, N:{self.N}")

    self.stats = Stats(strategy)

    ################ Heuristic Tiling #################
    tiling = strategy.tiling
    arr_mapping = strategy.arr_mapping
    loop_order = strategy.loop_order
    broadcast = strategy.broadcast
        
    arr_tile_M, arr_tile_N, arr_tile_K = find_arr_tile(self, pcb_module, strategy, debug)
    tile_size = find_tile_size(self, pcb_module, tiling, arr_tile_M, arr_tile_N, arr_tile_K, debug)
    M_tile = tile_size['M']
    N_tile = tile_size['N']
    K_tile = tile_size['K']

    M_t = self.M // M_tile
    N_t = self.N // N_tile
    K_t = self.K // K_tile
    M_remain = self.M % M_tile
    N_remain = self.N % N_tile
    K_remain = self.K % K_tile



    previous_m = 0
    previous_n = 0
    previous_k = 0
    total_latency = 0    
    total_io_latency = 0
    total_compute_latency = 0
    tile_latency = np.zeros(
        [ceil(self.M / M_tile), ceil(self.N / N_tile), ceil(self.K / K_tile)]
    )
    tile_MK_io_latency = np.zeros(
        [ceil(self.M / M_tile), ceil(self.N / N_tile), ceil(self.K / K_tile)]
    )
    tile_KN_io_latency = np.zeros(
        [ceil(self.M / M_tile), ceil(self.N / N_tile), ceil(self.K / K_tile)]
    )
    tile_MN_io_latency = np.zeros(
        [ceil(self.M / M_tile), ceil(self.N / N_tile), ceil(self.K / K_tile)]
    )
    tile_shape_M = np.zeros([ceil(self.M / M_tile), ceil(self.N / N_tile), ceil(self.K / K_tile)])
    tile_shape_N = np.zeros([ceil(self.M / M_tile), ceil(self.N / N_tile), ceil(self.K / K_tile)])
    tile_shape_K = np.zeros([ceil(self.M / M_tile), ceil(self.N / N_tile), ceil(self.K / K_tile)])
    if M_t * N_t * K_t != 0:
        tile_MK_io_latency[:M_t, :N_t, :K_t], tile_KN_io_latency[:M_t, :N_t, :K_t], tile_MN_io_latency[:M_t, :N_t, :K_t], tile_latency[:M_t, :N_t, :K_t] = get_tile_latency(self, pcb_module, strategy, {'M': M_tile, 'K': K_tile, 'N': N_tile}, debug)
        tile_shape_M[:M_t, :N_t, :K_t] = M_tile
        tile_shape_N[:M_t, :N_t, :K_t] = N_tile
        tile_shape_K[:M_t, :N_t, :K_t] = K_tile
    if M_remain != 0:
        tile_MK_io_latency[-1, :N_t, :K_t], tile_KN_io_latency[-1, :N_t, :K_t], tile_MN_io_latency[-1, :N_t, :K_t], tile_latency[-1, :N_t, :K_t] = get_tile_latency(self, pcb_module, strategy, {'M': M_remain, 'K': K_tile, 'N': N_tile}, debug)
        tile_shape_M[-1, :N_t, :K_t] = M_remain
        tile_shape_N[-1, :N_t, :K_t] = N_tile
        tile_shape_K[-1, :N_t, :K_t] = K_tile
    if N_remain != 0:
        tile_MK_io_latency[:M_t, -1, :K_t], tile_KN_io_latency[:M_t, -1, :K_t], tile_MN_io_latency[:M_t, -1, :K_t], tile_latency[:M_t, -1, :K_t] = get_tile_latency(self, pcb_module, strategy, {'M': M_tile, 'K': K_tile, 'N': N_remain}, debug)
        tile_shape_M[:M_t, -1, :K_t] = M_tile
        tile_shape_N[:M_t, -1, :K_t] = N_remain
        tile_shape_K[:M_t, -1, :K_t] = K_tile
    if K_remain != 0:
        tile_MK_io_latency[:M_t, :N_t, -1], tile_KN_io_latency[:M_t, :N_t, -1], tile_MN_io_latency[:M_t, :N_t, -1], tile_latency[:M_t, :N_t, -1] = get_tile_latency(self, pcb_module, strategy, {'M': M_tile, 'K': K_remain, 'N': N_tile}, debug)
        tile_shape_M[:M_t, :N_t, -1] = M_tile
        tile_shape_N[:M_t, :N_t, -1] = N_tile
        tile_shape_K[:M_t, :N_t, -1] = K_remain
    if M_remain * N_remain != 0:
        tile_MK_io_latency[-1, -1, :K_t], tile_KN_io_latency[-1, -1, :K_t], tile_MN_io_latency[-1, -1, :K_t], tile_latency[-1, -1, :K_t] = get_tile_latency(self, pcb_module, strategy, {'M': M_remain, 'K': K_tile, 'N': N_remain}, debug)
        tile_shape_M[-1, -1, :K_t] = M_remain
        tile_shape_N[-1, -1, :K_t] = N_remain
        tile_shape_K[-1, -1, :K_t] = K_tile
    if M_remain * K_remain != 0:
        tile_MK_io_latency[-1, :N_t, -1], tile_KN_io_latency[-1, :N_t, -1], tile_MN_io_latency[-1, :N_t, -1], tile_latency[-1, :N_t, -1] = get_tile_latency(self, pcb_module, strategy, {'M': M_remain, 'K': K_remain, 'N': N_tile}, debug)
        tile_shape_M[-1, :N_t, -1] = M_remain
        tile_shape_N[-1, :N_t, -1] = N_tile
        tile_shape_K[-1, :N_t, -1] = K_remain
    if N_remain * K_remain != 0:
        tile_MK_io_latency[:M_t, -1, -1], tile_KN_io_latency[:M_t, -1, -1], tile_MN_io_latency[:M_t, -1, -1], tile_latency[:M_t, -1, -1] = get_tile_latency(self, pcb_module, strategy, {'M': M_tile, 'K': K_remain, 'N': N_remain}, debug)
        tile_shape_M[:M_t, -1, -1] = M_tile
        tile_shape_N[:M_t, -1, -1] = N_remain
        tile_shape_K[:M_t, -1, -1] = K_remain
    if M_remain * N_remain * K_remain != 0:
        tile_MK_io_latency[-1, -1, -1], tile_KN_io_latency[-1, -1, -1], tile_MN_io_latency[-1, -1, -1], tile_latency[-1, -1, -1] = get_tile_latency(self, pcb_module, strategy, {'M': M_remain, 'K': K_remain, 'N': N_remain}, debug)
        tile_shape_M[-1, -1, -1] = M_remain
        tile_shape_N[-1, -1, -1] = N_remain
        tile_shape_K[-1, -1, -1] = K_remain
    # print(tile_latency)
    # with np.printoptions(threshold=np.inf):
    #     print(tile_latency)
    # mapping: 
    # M -> arrays
    # N -> banks
    # K -> devices
    # we assume for each tile's execution:
    # M_tilexK_tile is moved from somewhere else to compute-enabled DRAM and duplicated across banks to enable N tiling
    # K_tilexN_tile is already in compute-enabled DRAM and already duplicated across arrays to enable M tiling
    # M_tilexN_tile is moved from somewhere else to compute-enabled DRAM, does not need any duplication across K dimension (mapped to devices) as K is reduction axis

    ################ Simulate Loops #################
    for m, n, k in self.generate_tile_loops(
        ceil(self.M / M_tile),
        ceil(self.N / N_tile),
        ceil(self.K / K_tile),    
        loop_order,
    ):
        M_N_io_latency = tile_MN_io_latency[m,n,k]
        M_K_io_latency = tile_MK_io_latency[m,n,k]
        K_N_io_latency = tile_KN_io_latency[m,n,k]
        if m == 0 and n == 0 and k == 0:
            #load data for first tile
            total_latency += M_N_io_latency + M_K_io_latency + K_N_io_latency
            total_compute_latency += 0
            total_io_latency += M_N_io_latency + M_K_io_latency + K_N_io_latency
            continue


        # current tile read latency
        if m == previous_m and k == previous_k:
            current_tile_read_latency = K_N_io_latency
        elif n == previous_n and k == previous_k:
            current_tile_read_latency = M_K_io_latency
        else:
            current_tile_read_latency = (
                M_K_io_latency + K_N_io_latency
            )
        if k > 0 and not (m == previous_m and n == previous_n):
            current_tile_read_latency += M_N_io_latency
        
        # print(f"iter m{m} n{n} k{k}: double_buffering {double_buffering}")
        # previous tile compute latency
        previous_tile_compute_latency = tile_latency[previous_m, previous_n, previous_k]
        # if k > 0:
        #     previous_tile_compute_cycle_count += (
        #         previous_l2_tile.K_reduction_cycle_count
        #     )
        # previous tile write latency
        if m == previous_m and n == previous_n:
            previous_tile_write_latency = 0
        else:
            previous_tile_write_latency = M_N_io_latency

        double_buffering = False
        # read current tile, compute previous tile, write previous tile
        if double_buffering:  # pipelined
            total_latency += (
                max(
                    current_tile_read_latency, previous_tile_compute_latency
                )
                + previous_tile_write_latency
            )
        else:  # non-pipelined
            total_latency += (
                current_tile_read_latency
                + previous_tile_compute_latency
                + previous_tile_write_latency
            )
            total_io_latency += (current_tile_read_latency + previous_tile_write_latency)
            total_compute_latency += previous_tile_compute_latency
            # print(total_latency)
        previous_m = m
        previous_n = n
        previous_k = k

    # compute and write last tile
    total_latency += (
        tile_MN_io_latency[-1, -1, -1] #last tile write
        + tile_latency[-1, -1, -1] #lsat tile compute
    )
    total_io_latency += tile_MN_io_latency[-1, -1, -1] #last tile write
    total_compute_latency += tile_latency[-1, -1, -1] #lsat tile compute
    

    # if previous_k > 0:
    #     total_cycle_count += ceil(l2_tiles[-1, -1, -1].K_reduction_cycle_count)
    # if debug:
    #     print(f"gemm latency: {total_latency}, compute_latency: {total_compute_latency}, io_latency:{total_io_latency}")
    self.stats.latency = total_latency
    self.stats.compute_latency = total_compute_latency
    self.stats.io_latency = total_io_latency

    if debug:
        external_debug_info = {
            "M_remain": M_remain,
            "K_remain": K_remain,
            "N_remain": N_remain,
            "M_t": M_t,
            "K_t": K_t,
            "N_t": N_t
        }

        debug_info = self.stats.to_string(debug=True, debug_info=external_debug_info)

        print(debug_info)
    return total_latency

def compile_and_simulate_simdram(
    self,
    pcb_module: Device,
    strategy: TilingStrategy,
    debug: bool
):

    # debug = False
    assert pcb_module.type == "simdram"
    return heuristic_simdram_broadcast(self, pcb_module = pcb_module, strategy=strategy, debug=debug)
    
def compile_and_simulate(
    self,
    pcb_module: Device,
    compile_mode: str = "exhaustive",
    debug: bool = False,
):
    assert pcb_module.type == 'simdram'
    return compile_and_simulate_simdram(self,pcb_module, compile_mode, debug)