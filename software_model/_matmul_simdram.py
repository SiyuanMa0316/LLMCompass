from sympy import factor
from hardware_model.device import Device
from math import ceil, log2, floor
from software_model.utils import Tensor, DataType, TilingStrategy, simdram_op_latency_dict, simdram_PE_op_latency_dict
from software_model.utils import TilingStrategy, find_closest_divisor, Stats
import numpy as np
import math

def find_tile_K(self, row_limits : int, grow : int = 32) -> int:
    """_summary_
    Helper method to find the tile_K satisfy
    the per array storage
    Args:
        row_limits (int): row limits per array
        grow (int): default grow factor we used to estimate tile_K

    Returns:
        int: maximum power of 2 tile_K value 
    """
    k = 32
    # maximum tile_K without considering accumulation
    element_limit = row_limits // 2 // (self.data_type.word_size * 8) 
    # product_bits = 2 * self.data_type.word_size * 8
    accum_bits = ceil(log2(k)) + 2 * self.data_type.word_size * 8
    input_storage_bits = 2 * self.data_type.word_size * 8 * k
    while input_storage_bits + accum_bits < row_limits:
        k = k + grow
        accum_bits = ceil(log2(k)) + 2 * self.data_type.word_size * 8
        input_storage_bits = 2 * self.data_type.word_size * 8 * k
    k = k - grow
    print(f"K = {k}, accum_bits {accum_bits}, row_limits {row_limits}, element_limit {element_limit}")
    return k, accum_bits
    
def find_tile_N_M(self, col_limits : int, base : int = 16) -> tuple[int, int]:
    """helper method to find the suitable tile_N and tile_M
    if N and M are all not 1, return the nearest pow of 2 value of tile_N and tile_M
    otherwise, tile_N = 1 if N = 1 or tile_M =1 if M = 1

    Args:
        col_limits (int): columns per SIMDRAM array
        base (int): default guess of pow of 2
    Returns:
        tuple[int, int]: compted tile_N and tile_M
    """
    M = self.computational_graph.M
    N = self.computational_graph.N
    if M == 1:
        tile_M = 1
        tile_N = min(col_limits, N)
    if N == 1:
        tile_N = 1
        tile_M = min(col_limits, M)
    if M * N <= 128:
        tile_M = M
        tile_N = N

    M_is_larger = (M > N)
    
    if M_is_larger:
        tile_M = min(base, M)
        tile_N = col_limits // tile_M
    else:
        tile_N = min(base, N)
        tile_M = col_limits // tile_N
    assert tile_M * tile_N <= col_limits, f"tile_N and tile_M allocation failed N {N}, tile_N {tile_N}, M {M}, tile_M {tile_M}"
    
    return np.sort([tile_M, tile_N])

def get_tiling_factor(pcb_module: Device, tiling: dict):
    col_per_array = pcb_module.compute_module.bank.arr_cols
    row = pcb_module.compute_module.bank.arr_rows
    array_per_device_bank = pcb_module.compute_module.bank.arr_count
    bank = pcb_module.compute_module.bank_count
    device = pcb_module.compute_module.bank.device_count
    to_return = {'M': 1, 'K': 1, 'N': 1} 
    for key in tiling.keys():
        val = tiling[key]
        if val:
            for c in val:
                if c == 'A':
                    to_return[key] *= array_per_device_bank
                if c == 'B':
                    to_return[key] *= bank
                if c == 'D':
                    to_return[key] *= device
        else:
            # tiling[key] == None
            to_return[key] = 1
    return to_return


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
    
    # if 'A' in dup:
    #     # duplicate tile to every array
    #     temp = array_per_device_bank * tile_1 * tile_2 * word_size  / pcb_module.io_module.bandwidth
    #     if array_broadcast_enable:
    #         temp /= array_per_device_bank
    #     latency += temp

    # if 'B' in dup:
    #     # duplicate tile to every bank
    #     temp = bank * tile_1 * tile_2 * word_size / pcb_module.io_module.bandwidth
    #     if bank_broadcast_enable:
    #         temp /= bank
    #     latency += temp

    # if 'D' in dup:
    #     # duplicate tile to every device
    #     temp = device * tile_1 * tile_2 * word_size / pcb_module.io_module.bandwidth
    #     latency += temp
    
    # return latency

def simdram_gemv(self, pcb_module: Device, tilingStrategy: TilingStrategy, debug = True,) -> float:
    
    M = self.computational_graph.M
    N = self.computational_graph.N
    K = self.computational_graph.K

    tiling = tilingStrategy.tiling
    mapping = tilingStrategy.arr_mapping
    broad_cast = tilingStrategy.broadcast
    loop_order = tilingStrategy.loop_order
    pcb_module.compute_module.with_PE = tilingStrategy.with_PE

    assert M == 1, "DRAM PIM require input M = 1 or N = 1"

    col_per_array = pcb_module.compute_module.bank.arr_cols
    row = pcb_module.compute_module.bank.arr_rows
    array_per_device_bank = pcb_module.compute_module.bank.arr_count
    bank = pcb_module.compute_module.bank_count
    device = pcb_module.compute_module.bank.device_count
    
    pe_op_latency = simdram_PE_op_latency_dict[self.data_type.name]['add'] + simdram_PE_op_latency_dict[self.data_type.name]['mul']
    normal_op_latency = simdram_op_latency_dict[self.data_type.name]['add'] + simdram_op_latency_dict[self.data_type.name]['mul']
    mac_latency =  pe_op_latency if pcb_module.compute_module.with_PE else normal_op_latency

    row_element_storage = row // self.data_type.word_size // 8 // 2
    rank = 1
    pcb_module.io_module.bandwidth = 19.2 * (1024/1000) ** 3 # bandwidth in bytes per ns
    total_arrays = array_per_device_bank * bank * device * rank

    arr_tile_M = 1
    # default we only need intra-array-multicast
    popcorn_adder = False
    intra_array_multicast = True
    if mapping:
        k_arr_map = mapping['K']
        n_arr_map = mapping['N']
    else:
        # default map K to array columns and N to array rows
        k_arr_map = 'R'
        n_arr_map = 'C'
    
    if k_arr_map == 'C':
        popcorn_adder = True
        intra_array_multicast = False
    # smallest problem size to each array
    arr_tile_N = col_per_array if n_arr_map == 'C' else row_element_storage
    arr_tile_K = row_element_storage if k_arr_map == 'R' else col_per_array

    # extract tiling strategy
    tiling_factors = get_tiling_factor(pcb_module, tiling)
    tiling_factor_M = 1 # GEMV always assume 1 for dimension_M 
    tiling_factor_N = tiling_factors['N']
    tiling_factor_K = tiling_factors['K']

    tile_M = arr_tile_M * tiling_factor_M
    tile_K = arr_tile_K * tiling_factor_K
    tile_N = arr_tile_N * tiling_factor_N

    util_M = M / tile_M
    util_N = N / tile_N
    util_K = K / tile_K
    
    opt_arr_tile_M = arr_tile_M
    opt_arr_tile_K = arr_tile_K
    opt_arr_tile_N = arr_tile_N

    opt_tile_M = tile_M
    opt_tile_N = tile_N
    opt_tile_K = tile_K

    # optimize the arr_tile size to fully utilize the SIMDRAM
    if util_M < 1:
        opt_arr_tile_M = ceil(arr_tile_M * util_M)
        opt_tile_M = opt_arr_tile_M * tiling_factor_M
    if util_N < 1:
        opt_arr_tile_N = ceil(arr_tile_N * util_N)
        opt_tile_N = opt_arr_tile_N * tiling_factor_N
    if util_K < 1:
        opt_arr_tile_K = ceil(arr_tile_K * util_K)
        opt_tile_K = opt_arr_tile_K * tiling_factor_K

    row_util = opt_arr_tile_K / row_element_storage if k_arr_map == 'R' else opt_arr_tile_N / row_element_storage
    compute_util = opt_arr_tile_N / col_per_array if n_arr_map == 'C' else opt_arr_tile_K / col_per_array

    num_mac = row_util * row_element_storage
    compute_latency = num_mac * mac_latency

 

    dups = ['A', 'B', 'D']
    K_N_dup = []
    M_K_dup = []
    for c in dups:
        # if not tiled along A/B/D
        # we need to duplicate this tile to corresponding place
        if tiling['N']:
            if c not in tiling['N']:
                K_N_dup.append(c)
        if tiling['K']:
            if c not in tiling['K']:
                M_K_dup.append(c)
        if tiling['M']:
            if c not in tiling['M']:
                M_K_dup.append(c)



    M_K_io_latency = get_tile_io_latency(pcb_module, broad_cast, opt_tile_M, opt_tile_K, self.data_type.word_size, M_K_dup)
    K_N_io_latency = get_tile_io_latency(pcb_module, broad_cast, opt_tile_N, opt_tile_K, self.data_type.word_size, K_N_dup)
    # no duplication required for M_N tile, simply write it back to Host
    M_N_io_latency = opt_tile_M * opt_tile_N * self.data_type.word_size / pcb_module.io_module.bandwidth
    # assume K_N tile is inside SIMDRAM
    K_N_io_latency = 0
    
    # simulate complete end-to-end GEMV latency in tile basis 
    num_tile_K = K // opt_tile_K
    remain_K = K % opt_tile_K

    num_tile_N = N // opt_tile_N
    remain_N = N % opt_tile_N

    prev_m = 0
    prev_n = 0
    prev_k = 0
    total_latency = 0

    for m, k, n in self.generate_tile_loops(1, num_tile_N, num_tile_K, loop_order=loop_order):
        # load first tile
        if m == 0 and n == 0  and k == 0:
            total_latency += M_K_io_latency + K_N_io_latency
            if num_tile_K == num_tile_N == 1: # if only one tile for both N and K, we have to add the compute latency
                total_latency += compute_latency
            continue
        
        current_tile_io_latency = 0
        if m == prev_m and k == prev_k:
            # load new tile_N_K
            current_tile_io_latency += K_N_io_latency
        if n == prev_n and k == prev_k:
            # load new tile_M_K
            current_tile_io_latency += M_K_io_latency
        if not(m == prev_m and n == prev_n):
            # either M or N move to next tile
            # and we have at least 1 M_tile result available
            current_tile_io_latency += M_N_io_latency

        total_latency += current_tile_io_latency + compute_latency
    
    # remaining tiles: 1 x remain_K, remain_K x remain_N

    remain_M_K_io_latency = get_tile_io_latency(pcb_module, broad_cast, opt_tile_M, remain_K, self.data_type.word_size, M_K_dup)
    remain_K_N_io_latency = get_tile_io_latency(pcb_module, broad_cast, remain_K, remain_N, self.data_type.word_size, K_N_dup)
    remain_row_util = remain_K / row_element_storage if k_arr_map == 'R' else remain_N / row_element_storage
    
    remain_mac_latency = remain_row_util * compute_latency * row_element_storage

    remain_M_N_io_latency = opt_tile_M * remain_N * self.data_type.word_size / pcb_module.io_module.bandwidth


    total_latency += remain_K_N_io_latency + remain_M_K_io_latency + remain_M_N_io_latency + remain_mac_latency

    if debug:
        print(f"{'Parameter':<30}{'Value':<20}{'Parameter':<30}{'Value':<20}{'Parameter':<30}{'Value':<20}")
        print(f"{'-'*150}")
        print(f"{'arr-tile-M':<30}{arr_tile_M:<20}{'arr-tile-K':<30}{arr_tile_K:<20}{'arr-tile-N':<30}{arr_tile_N:<20}")
        print(f"{'tiling_factor_M':<30}{tiling_factor_M:<20}{'tiling_factor_K':<30}{tiling_factor_K:<20}{'tiling_factor_N':<30}{tiling_factor_N:<20}")
        print(f"{'tile_M':<30}{tile_M:<20}{'tile_K':<30}{tile_K:<20}{'tile_N':<30}{tile_N:<20}")
        print(f"{'opt_arr_tile_M':<30}{opt_arr_tile_M:<20}{'opt_arr_tile_K':<30}{opt_arr_tile_K:<20}{'opt_arr_tile_N':<30}{opt_arr_tile_N:<20}")
        print(f"{'opt_tile_M':<30}{opt_tile_M:<20}{'opt_tile_K':<30}{opt_tile_K:<20}{'opt_tile_N':<30}{opt_tile_N:<20}")
        print(f"{'util_M':<30}{util_M * 100:.2f}%{'':<15}{'util_K':<30}{util_K * 100:.2f}%{'':<15}{'util_N':<30}{util_N * 100:.2f}%")
        print(f"{'Row Capacity Utilization':<30}{row_util * 100:.2f}%{'':<15}{'SIMD Utilization':<30}{compute_util * 100:.2f}%")
        print(f"{'Intra_array_multicast':<30}{str(bool(intra_array_multicast)):<20}{'Popcorn adder':<30}{str(bool(popcorn_adder)):<20}{'Using Bit-serial PE':<30}{str(bool(pcb_module.compute_module.with_PE)):<20}")
        print(f"{'Compute Latency':<30}{compute_latency * 1e-6:.3f}ms{'':<15}{'M_K_io_latency':<30}{M_K_io_latency * 1e-6:.3f}ms{'':<15}{'K_N_io_latency':<30}{K_N_io_latency * 1e-6:.3f}ms")
        print(f"{'Remain Compute Latency':<30}{remain_mac_latency * 1e-6:.3f}ms{'':<15}{'remain_K_N_io_latency':<30}{remain_K_N_io_latency * 1e-6:.3f}ms{'':<15}{'remain_M_N_io_latency':<30}{remain_M_N_io_latency * 1e-6:.3f}ms")
        print(f"{'Total Latency':<30}{total_latency * 1e-6:.3f}ms")




    return total_latency




def simdram_gemv_broadcast_only(self, pcb_module: Device, debug = False) -> float:
    M = self.computational_graph.M
    N = self.computational_graph.N
    K = self.computational_graph.K

    assert N == 1 or M == 1, f"SIMDRAM_GEMV_V2 require input M = 1 or N = 1"
    col_per_array = pcb_module.compute_module.bank.arr_cols
    row = pcb_module.compute_module.bank.arr_rows
    array_per_device_bank = pcb_module.compute_module.bank.arr_count
    bank = pcb_module.compute_module.bank_count
    device = pcb_module.compute_module.bank.device_count

    rank = 1
    pcb_module.io_module.bandwidth = 19.2 * (1024/1000) ** 3 # bandwidth in bits per ns
    total_arrays = array_per_device_bank * bank * device * rank
    '''
    Assume input M = 1, LHS is a row vector P and RHS is a complete matrix Q.
    For M, K, N, K has the largest number.
    Dimension partition: 
    M = 1, no partition. 
    N partition across Device and Bank level.  
    K partition across Array.
    Therefore, we will need cross-array reduction.
    '''
    tile_M = 1
    factor_M = 1
    remain_M = M % (factor_M * tile_M)    

    tile_N = col_per_array / 2
    factor_N = floor(N / tile_N)
    remain_N = N % tile_N

    # tile_K, accum_bits = find_tile_K(self, row)
    # factor_K= floor(K / tile_K)
    # remain_K = K % tile_K
    
    # use every array in every bank and every device
    factor_K = array_per_device_bank   
    tile_K = K // factor_K
    tile_K = tile_K // 2
    factor_K = K // tile_K
    remain_K = K % tile_K

    # As we are duplicating every tile of LHS vector to exact array of every bank and device
    # and we have broadcasting hardware across bank level, we dont need duplication across bank-level.
    # the M_K_IO_latency = factor_K * tile_K * data_width * device/ BW
    # Meanwhile, we dont need to duplication RHS,
    # so, K_N_IO_latency = factor_K * factor_N * tile_K * tile_N * data_width / BW
    major_M_K_IO_latency = factor_K * tile_K * self.data_type.word_size * device/ pcb_module.io_module.bandwidth
    major_K_N_IO_latency = factor_K * factor_N * tile_K * tile_N * self.data_type.word_size / pcb_module.io_module.bandwidth
    mac_latency = tile_K * (simdram_PE_op_latency_dict[self.data_type.name]['mul'] + simdram_PE_op_latency_dict[self.data_type.name]['add'])
    # mac_latency = tile_K * (simdram_op_latency_dict[self.data_type.name]['mul'] + simdram_op_latency_dict[self.data_type.name]['add'])

    major_IO_latency =  major_K_N_IO_latency + major_M_K_IO_latency

    '''
    After complete SIMDRAM launch, we have 2 remaining dimensions: remain_K and remain_N
    LHS remains: M x remain_K = 1 x remain_K, RHS remains 1 row vector (remain_K x N) and 1 column vector (K x remain_N)
    and they have a small overlapping matrix of (remain_K x remain_N). 
    Lets ignore this overlapping matrix for now, and focus only on the (N - remain_N) and (K - remain_K).
    
    For remained row vector of RHS, there are factor_N x (remain_K x tile_N) matrices, and each of them will
    have to multiply with the last (1 x remain_K) row vector of LHS. Total factor_N x [(1, remain_K) x (remain_K, tile_K)] GEMV. 

    Similarly, remained column vector of RHS, there are factor_K x (tile_K x remain_N) matrices, and each of them 
    will have to multiply with every (1 x tile_K) row vector of LHS. Total factor_K x [(1, tile_K) x (tile_K, remain_N)] GEMVs.

    lastly, we still have a (1 x remain_K) x (remain_K x remain_N) GEMV.
    '''
    # to compute factor_N x (1 x remain_K) x (remain_K x tile_N)
    # we need to duplicate the (1 x remain_K) across device and bank, following same broadcasting scheme as above.
    remain_M_K_IO_latency = tile_K * self.data_type.word_size * device / pcb_module.io_module.bandwidth
    # to compute factor_K x (1 x tile_K) x (tile_K x tile_N) and last (1 x remain_K) x (remain_K x remain_N)
    # we dont need duplication
    remain_K_N_IO_latency = (M * K + K * remain_N) * self.data_type.word_size / pcb_module.io_module.bandwidth
    remain_IO_latency = remain_K_N_IO_latency + remain_M_K_IO_latency

    iteration_K = factor_K / array_per_device_bank
    iteration_N = factor_N / bank / device
    arrays_for_major_tiles = factor_K * factor_N
    arrays_for_remaining_tiles = factor_N + factor_K

    if debug:
        print(f"SIMDRAM GEMV Broadcast\nM = {M} = {tile_M} * {factor_M} + {remain_M}\nK = {K} = {tile_K} * {factor_K} + {remain_K}\nN = {N} = {tile_N} * {factor_N} + {remain_N}")
        print(f"Partition K -> array: K_iteration {iteration_K}, partition N -> bank and device: N_iteration {iteration_N}")
        print(f"Array required for major tiles :{arrays_for_major_tiles}, Array required for remaining tiles: {arrays_for_remaining_tiles},Total Arrays: {total_arrays}")
        print(f"Major tiles IO latency = {major_IO_latency * 1e-6}ms, Remain tiles IO latency = {remain_IO_latency * 1e-6}ms Mac Latency = {mac_latency * 1e-6}ms")


    total_latency = 0
    if total_arrays > (arrays_for_major_tiles + arrays_for_remaining_tiles):
        total_latency = mac_latency + major_IO_latency + remain_IO_latency
    else:
        # cannot schedule remain tiles along with major tiles together
        print(f"Input GEMV size is too large, cannot schedule remain tiles with major tiles together.")

    return total_latency
    




def simdram_heuristic_tiling_v2(self, pcb_module: Device,  tilingStrategy: TilingStrategy, debug=False) -> float:

    pcb_module.io_module.bandwidth = 19.2 * (1024/1000) ** 3 # bandwidth in bytes per ns
    M = self.computational_graph.M
    N = self.computational_graph.N
    K = self.computational_graph.K


    if M == 1 or N == 1:
        return simdram_gemv(self,pcb_module, tilingStrategy=tilingStrategy, debug=debug)

    M_K_bits = M * K * self.data_type.word_size * 8
    K_N_bits = K * N * self.data_type.word_size * 8
    
    # channels, bank, row size
    if debug:
        print(f"Heuristic-SIMDRAM Tiling Simulation: M {self.M}, K {self.K}, N {self.N}")
    """
    Hierarchy-Level: Rank -> device = bank -> array
    """
    col_per_array = pcb_module.compute_module.bank.arr_cols
    row = pcb_module.compute_module.bank.arr_rows
    array_per_device_bank = pcb_module.compute_module.bank.arr_count
    bank = pcb_module.compute_module.bank_count
    device = pcb_module.compute_module.bank.device_count
    rank = 1
    capacity_per_array = col_per_array * row
    capacity_per_bank = capacity_per_array * array_per_device_bank
    capacity_per_device = capacity_per_bank * bank
    capacity_per_rank = capacity_per_device * device
    total_capacity = capacity_per_rank * rank

    # print(f"Input GEMM Storage Size: Input + Output {(M_K_bits + K_N_bits + M_N_bits)/1024/1024/1024/8}GB")
    # print(f"capacity_per_array:{capacity_per_array/1024/1024/8}MB\n"+ 
    #     f"Capacity_per_bank:{capacity_per_bank/1024/1024/8}MB\n"+ 
    #     f"Capacity_per_device:{capacity_per_device/1024/1024/1024/8}GB\n"+
    #     f"Capacity_per_rank:{capacity_per_rank/1024/1024/1024/8}GB\n"+
    #     f"Total_capacity:{total_capacity/1024/1024/1024/8}GB")
    
    
    """
    Data Layout in single array
    A = tile_M x tile_K = [[1 2 3 4], [5 6 7 8], [9 10 11 12]]
    B = tile_K x tile_N = [[A A'], [B B'], [C C'], [D D']]
    C = tile_M x tile_N = [[C1 C2], [C3 C4], [C5 C6]]
    A x B --->  1  2  3  4      A A'
                5  6  7  8      B B' 
                9  10 11 12     C C'
                                D D'
    Layout in bit-serial operation format 
    Input  A        A   A   A   A'  A'  A'  total tile_N * tile_M columns
                    B   B   B   B'  B'  B'  requires tile_N * tile_M <= col_per_array 
                    C   C   C   C'  C'  C'  
                    D   D   D   D'  D'  D'  
    Input B         1   5   9   1   5   9   
                    2   6   10  2   6   10   
                    3   7   11  3   7   11
                    4   8   12  4   8   12
    Result C       C1   C2  C3  C4  C5  C6  

    We need K rows per input matrix, total 2 * K rows for input storage
    We also need to reserve result for accumulation 
    product bits = 2*self.data_type.word_size * 8, 
    accumulation requires 2 * K product to be reduced at the same time
    accum_bits =  ceil(log2(2 * K) + product_bits) = product_bits + 1 + ceil(log2(K))

    Total row storage requires: 2 * self.data_type.word_size * 8 * 2 * K + accum_bits <= row
    """
    #find nearest power of 2 of tile_N, tile_M that satisfy the col_per_array limit
    tile_N, tile_M = find_tile_N_M(self, col_per_array)
    factor_N = ceil(N / tile_N)
    factor_M = ceil(M / tile_M)
    # find nearest power of 2 of tile_K that satisfy the row limit
    tile_K, accum_bits =find_tile_K(self, row)
    factor_K = floor(K / tile_K)
    remain_K = K - factor_K * tile_K

    M_N_bits = M * N * accum_bits * 8
    """
    A = M x K = factor_M x factor_K x (tile_M x tile_K)
    B = K x N = factor_K x factor_N x (tile_K x tile_N)
    #multiplications per column = #additions per column 
    Since all columns are operated in parallel, and each column
    can only perform computation at 1 time. 
    total #Mul = K, #Add = K
    """
    product_bits = self.data_type.word_size * 8 * 2
    input_storage_bits = 2 * tile_K * self.data_type.word_size * 8
    col_fragmentation_per_array = tile_N * tile_M / col_per_array
    row_fragmentation_per_array = (input_storage_bits + product_bits + accum_bits) / row

    assert tile_N * tile_M <= col_per_array, "column storage allocation exceed the array limit"
    assert input_storage_bits + product_bits + accum_bits < row, f"row storage allocation exceed the array limit: tile_K {tile_K} remain_K {remain_K}"

    if debug:
        print(f"A = factor_M {factor_M} x factor_K {factor_K} x (tile_M {tile_M} x tile_K {tile_K})")
        print(f"B = factor_K {factor_K} x factor_N {factor_N} x (tile_K {tile_K} x tile_N {tile_N})")
        print(f"Remain workloads tile_M {tile_M} x remain_K {remain_K} x tile_N {tile_N} Accumulation BitWidth {accum_bits}")

        print(f"column utilization per array:{col_fragmentation_per_array * 100:.2f}%")
        print(f"row utilization per array:{row_fragmentation_per_array * 100:.2f}%")

    #per array latency
    add_latency_per_array = tile_K * simdram_op_latency_dict[self.data_type.name]['add']
    mul_latency_per_array = tile_K * simdram_op_latency_dict[self.data_type.name]['mul']
    # Add extra accumulation latency
    compute_latency_per_array = add_latency_per_array + mul_latency_per_array + simdram_op_latency_dict['fp32']['add']
    if debug:
        print(f"Compute Latency per array {compute_latency_per_array} ns")
    """
    Bigger-tile view of matrix A and B
    A = factor_M x factor_K x (tile_M, tile_K), 
    B = factor_K x factor_N x (tile_K, tile_N)
    For A: (tile_M, tile_K) is the smallest unit of workload
    For B: (tile_K, tile_N) is the smallest unit of workload

    A = A(1,1)          A(1,2)          A(1,3)          ...     A(1,factor_K)
        A(2,1)          A(2,2)          A(2,3)          ...     A(2,factor_K)
        ...
        A(factor_M,1)   A(factor_M,2)   A(factor_M,3)   ...     A(factor_M, factor_K)

    B = B(1,1)          B(1,2)          B(1,3)          ...     B(1,factor_N)
        B(2,1)          B(2,2)          B(2,3)          ...     B(2,factor_N)
        ...
        B(factor_K,1)   B(factor_K,2)   B(factor_K,3)   ...     B(factor_K,factor_N)

    Vectorization GEMM in bigger-tile view
        A(1,1)          A(1,2)          ...          A(1,factor_K)         
    [   A(2,1)          A(2,2)          ...          A(2,factor_K)          ]   *     [B(1,1) B(2,1) ... B(factor_K,1)].T 
        ...
        A(factor_M,1)   A(factor_M,2)   ...         A(factor_M,factor_K)

    is equivalent to following Vector-Tile operations
        A(1,1)                                     A(1,2)                                   A(1,factor_K)
    [   A(2,1)          ] * [B(1,1)]   +    [      A(2,2)      ]  * [B(2,1)] + ... +    [   A(2,factor_K)   ] * [B(factor_K, 1)]
        ...                                        ...                                      ...
        A(factor_M,1)                              A(factor_M,2)                            A(factor_M,factor_K)

    For every smallest-unit workload of A and B, they have to take a full array for computation as calculated in previous stage determining the 
    tile size of M, K and N. 
    Re-call the SIMDRAM arch-specs, Rank -> device = bank -> array, 
    
    For each array: 
        Every array will have same bigger-tile B within same device,
        partition complete A into each array at batch size of 'array' along the M-dimension (downwards from A(1,1) to A(factor_M,1)).
        iterations in each array is determined by: iteration_array = ceil(factor_M / array).
        
        every array will output col_per_array * accum_bits bits 

    """
    iteration_array = ceil(factor_M / array_per_device_bank) # partition factor_M to every array
    output_bits_per_array = col_per_array * accum_bits
    """
    For each device: 
        Every device will have different bigger-tile B from assigned column vector,
        partition column vector of B into each device at batch size of 'device' along the K-dimension (downwards from B(1,1) to B(factor_K,1)). 
        iteration of each bank is determined by: iteration_device = ceil(factor_K / device).
    """
    # map each (tile_K, tile_N) along dimension k to every device, every array within same device will have same (tile_K, tile_N)
    iteration_device = ceil(factor_K / device) 
    # map K dimension to each bank
    iteration_bank = ceil(factor_N / bank)
    MK_bits_duplicated = M_K_bits * bank
    KN_bits_duplicate = K_N_bits * array_per_device_bank
    MN_bits_duplicated = M_N_bits * device
    
    """
    To further utilize the parallelism, we consider double tiling
    """
    double_tilingK = False
    double_tilingN = False
    iteration_double_tilingK = ceil(factor_K / device / bank) * factor_N
    iteration_double_tilingN = ceil(factor_N / device / bank) * factor_K
    
    if iteration_double_tilingN < (iteration_bank * iteration_device) and iteration_double_tilingN < iteration_double_tilingK:
        double_tilingN = True
        iteration_device = factor_K
        iteration_bank = ceil(factor_N / device / bank)
        MK_bits_duplicated = M_K_bits * device * bank
        MN_bits_duplicated = M_N_bits
    
    if iteration_double_tilingK < (iteration_bank * iteration_device) and iteration_double_tilingK < iteration_double_tilingN:
        double_tilingK = True
        iteration_device = ceil(factor_K / device / bank)
        iteration_bank = factor_N
        MK_bits_duplicated = M_K_bits
        MN_bits_duplicated = M_N_bits * device * bank

    double_tiling = True if double_tilingK or double_tilingN else False

    if double_tiling and debug:
        tile_str = f"Double Tiling along K-dimension" if double_tilingK else "Double Tiling along N-dimension"
        print(tile_str + f" array_iteration {iteration_array} x  bank_iteration {iteration_bank} x device_iteration {iteration_device}")


    # have to compare the utilization of device and bank


    """    
    For each bank:
        We assume only 1 rank for now, no parallelism across rank. However, we do have to consider the total data transfer latency
        and also if it is possible to schedule pipeline computation.
    """

    # input matrix needs to be shuffled and duplicated. This is done by reading out the input matrix to host, shuffle in host, then write back the duplicated matrix
    # currently we don't model the different available bandwidth of data read/write (for example, reading from a single bank only can cause lower read bandwidth)
    # we also don't model the host processing as we assume the memory bandwidth is always bottleneck
    input_shuffle_dup_latency = M_K_bits + MK_bits_duplicated / pcb_module.io_module.bandwidth / 8

    # if there is tiling on K across device/array/bank, then reduction across device/array/bank needs to be performed by host
    # host first reads KN_bits_duplicate bits, then reduce them to K_N_bits bits and write back
    # we also don't model the host processing as we assume the memory bandwidth is always bottleneck
    reduction_writeback_latency = KN_bits_duplicate / pcb_module.io_module.bandwidth / 8 + K_N_bits / pcb_module.io_module.bandwidth / 8 

    # total_output_bits = output_bits_per_array * array_per_device_bank * device * bank * iteration_array * iteration_bank * iteration_device
    output_bits_per_device_iteration = output_bits_per_array * array_per_device_bank
    output_bits_per_bank_iteration = output_bits_per_device_iteration * device
    output_bits_per_rank_iteration = output_bits_per_bank_iteration * bank
    total_output_bits = output_bits_per_rank_iteration * iteration_array * iteration_bank * iteration_device

    # print(f"output_bits_per_array: {output_bits_per_array}, Output bits per device per iteration: {output_bits_per_array * array_per_device_bank}\n"\
    # f"output bits per bank per iteration: {output_bits_per_bank_iteration} output bits per rank per iteration {output_bits_per_rank_iteration}\n"\
    # f"Total output bits {total_output_bits} iteration_array {iteration_array} iteration_device {iteration_device} iteration_bank {iteration_bank} IO_bandwidth {pcb_module.io_module.bandwidth} bits/ns\n")
    
    write_back_latency =  total_output_bits / pcb_module.io_module.bandwidth
    compute_latency = compute_latency_per_array * iteration_array * iteration_bank * iteration_device

    latency = compute_latency + write_back_latency
    # data amplification due to the vectorization of GEMM
    # each tile of B is duplicated factor_M x factor_N x factor_K times
    amplification_bits = factor_M * factor_N * factor_K * (tile_K * tile_N) * self.data_type.word_size * 8

    """
    We need to process the remaining workloads
    remain_A = factor_M x tile_M x remain_K , 
    remain_B = remain_K x tile_N x factor_N
    
    We still follow the same procedure to schedule the workload across the device and bank
    
    Based on previous calculation, the remaining smallest workload size (tile_M x remain_K, remain_K x tile_N) is guaranteed to fit within single array. 
    For simplicity, no partition across device is performed, and we use only 1 device per bank. 
    
    1. Scatter each remaining B tiles into arrays across the device: every array has same tile B but different tile A.
    remain_iteration_array = ceil(factor_M / array)
    each array have overlapped latency and same total ops #adds = #muls = remain_K
    
    2. B tiles are partitioned across banks (following same procedure as before: partition N along bank)
    remain_iteration_array = ceil(factor_N / bank)
    each bank have overlapped latency and same total ops #adds = #muls = remain_K * array
    
    __remain_compute_latency__: remain_iteration_array * remain_compute_latency_per_array
    __remain_bit_width__: same output bits each array as normal schedule. 
    Total bits output for every bank = (remain_array_factor * remain_bank_factor) * output_bits_per_array * array

    __remain_write_back_latency__: remain_iteration_array * remain_data_width /(bank_bandwidth)
    """
    # map the (tile_M, remain_K) to each array
    remain_iteration_array = ceil(factor_M / array_per_device_bank)
    # copy (remain_K, tile_N) along N dimension to each bank, every array within same bank has same (remain_K, tile_N)
    remain_iteration_bank = ceil(factor_N / bank / device)
    
    remain_add_latency_per_array = remain_K * simdram_op_latency_dict[self.data_type.name]['add']
    remain_mul_latency_per_array= remain_K * simdram_op_latency_dict[self.data_type.name]['mul']
    

    remain_compute_latency_per_array = (remain_add_latency_per_array  + remain_mul_latency_per_array)

    remain_compute_latency = remain_compute_latency_per_array * remain_iteration_array * remain_iteration_bank
    print(f"remain_compute_latency: {remain_compute_latency_per_array} ns, {remain_iteration_array}, {remain_iteration_bank}")
    
    # required computation for remained workloads

    remain_output_bits_per_bank = remain_iteration_array * output_bits_per_array * array_per_device_bank
    remain_write_back_latency = remain_output_bits_per_bank * bank * remain_iteration_bank / pcb_module.io_module.bandwidth

    remain_latency = remain_write_back_latency + remain_compute_latency
    

    amplification_bits = amplification_bits + self.data_type.word_size * 8 * (tile_N * remain_K) * factor_N * factor_M

    total_compute_latency = remain_compute_latency + compute_latency
    total_write_back_latency = remain_write_back_latency + write_back_latency

    total_latency = remain_latency + latency
    if debug:
        print(f"\n------- SIMDRAM Heuristic Tiling V2 Results --------")
        print(f"total latency {total_latency} ns, total_compute_latency {total_compute_latency} ns, total_write_back_latency {total_write_back_latency} ns")
        print(f"major_compute_latency {compute_latency} ns, major_write_back_latency {write_back_latency} ns")
        print(f"remain_compute_latency {remain_compute_latency}ns, remain_write_back_latency {remain_write_back_latency} ns\n")
    return total_latency

def heuristic_simdram(self, pcb_module):
    '''
    simdram gemm modelling with heuristic tiling and non broadcasting
    '''
    print(f"  matmul: {self.input1_shape}, {self.input2_shape}, {self.output_shape}")
    print(f"  M:{self.M}, K:{self.K}, N:{self.N}")

    num_col_per_array = pcb_module.compute_module.bank.arr_cols
    num_row = pcb_module.compute_module.bank.arr_rows
    num_array = pcb_module.compute_module.bank.arr_count
    num_bank = pcb_module.compute_module.bank_count
    num_device = pcb_module.compute_module.bank.device_count
    num_rank = 1
    capacity_per_array = num_col_per_array * num_row
    capacity_per_bank = capacity_per_array * num_array
    capacity_per_device = capacity_per_bank * num_bank
    capacity_per_rank = capacity_per_device * num_device
    total_capacity = capacity_per_rank * num_rank


    arr_tile_N, arr_tile_M = find_tile_N_M(self, num_col_per_array)
    factor_N = ceil(self.N / arr_tile_N)
    factor_M = ceil(self.M / arr_tile_M)
    # find nearest power of 2 of tile_K that satisfy the row limit
    arr_tile_K, accum_bits = find_tile_K(self, num_row)
    factor_K = floor(self.K / arr_tile_K)
    print(f"arr_tile_M:{arr_tile_M}, arr_tile_N:{arr_tile_N}, arr_tile_K:{arr_tile_K}")
    #heuristic tiling M to array, N to bank, K to device
    M_tile = arr_tile_M * num_array
    N_tile = arr_tile_N * num_bank
    K_tile = arr_tile_K * num_device
    
    print(f"tile size: {M_tile}, {K_tile}, {N_tile}")
    simdram_loop_order = "nkm"
    previous_m = 0
    previous_n = 0
    previous_k = 0
    total_latency = 0

    K_N_io_latency = K_tile * N_tile * self.data_type.word_size * num_array / pcb_module.compute_module.bandwidth #move K_tilexN_tile to compute-dram, consider duplication across arrays
    M_K_io_latency = M_tile * K_tile * self.data_type.word_size * num_bank / pcb_module.compute_module.bandwidth #move K_tilexK_tile to compute-dram, consider duplication across banks
    M_N_io_latency = M_tile * N_tile * self.data_type.word_size / pcb_module.compute_module.bandwidth #move M_tilexN_tile to compute-dram, no duplication as K is reduction axis

    M_t = self.M // M_tile
    N_t = self.N // N_tile
    K_t = self.K // K_tile
    M_remain = self.M % M_tile
    N_remain = self.N % N_tile
    K_remain = self.K % K_tile
    arr_K_remain = ceil(K_remain / num_device)

    #per array latency
    add_latency_per_array = arr_tile_K * simdram_op_latency_dict[self.data_type.name]['add']
    mul_latency_per_array = arr_tile_K * simdram_op_latency_dict[self.data_type.name]['mul']
    # Add extra accumulation latency
    tile_compute_latency = (add_latency_per_array + mul_latency_per_array + simdram_op_latency_dict['fp32']['add'] + simdram_op_latency_dict['fp32']['mul'])*1e-9
    # reduction across devices
    tile_compute_latency += M_tile * N_tile * num_device * self.data_type.word_size / pcb_module.compute_module.bandwidth

    #per array latency for K_remain
    add_latency_per_array_remain = arr_K_remain * simdram_op_latency_dict[self.data_type.name]['add']
    mul_latency_per_array_remain = arr_K_remain * simdram_op_latency_dict[self.data_type.name]['mul']
    # Add extra accumulation latency
    tile_compute_latency_remain = (add_latency_per_array_remain + mul_latency_per_array_remain + simdram_op_latency_dict['fp32']['add'] + simdram_op_latency_dict['fp32']['mul'])*1e-9
    # reduction across devices
    tile_compute_latency += M_remain * N_remain * num_device * self.data_type.word_size / pcb_module.compute_module.bandwidth
    
    print(f"K_N_io_latency: {K_N_io_latency}, M_K_io_latency: {M_K_io_latency}, M_N_io_latency: {M_N_io_latency}, tile_compute_latency:{tile_compute_latency}, tile_compute_latency_remain:{tile_compute_latency_remain}")

    tile_latency = np.zeros(
        [ceil(self.M / M_tile), ceil(self.N / N_tile), ceil(self.K / K_tile)]
    )
    if M_t * N_t * K_t != 0:
        tile_latency[:M_t, :N_t, :K_t] = tile_compute_latency
    if M_remain != 0:
        tile_RKN_compute_latency = tile_compute_latency
        tile_latency[-1, :N_t, :K_t] = tile_RKN_compute_latency
    if N_remain != 0:
        tile_MKR_compute_latency = tile_compute_latency
        tile_latency[:M_t, -1, :K_t] = tile_MKR_compute_latency
    if K_remain != 0:
        tile_MRN_compute_latency = tile_compute_latency_remain
        tile_latency[:M_t, :N_t, -1] = tile_MRN_compute_latency
    if M_remain * N_remain != 0:
        tile_RKR_compute_latency = tile_compute_latency
        tile_latency[-1, -1, :K_t] = tile_RKR_compute_latency
    if M_remain * K_remain != 0:
        tile_RRN_compute_latency = tile_compute_latency_remain
        tile_latency[-1, :N_t, -1] = tile_RRN_compute_latency
    if N_remain * K_remain != 0:
        tile_MRR_compute_latency  = tile_compute_latency_remain
        tile_latency[:M_t, -1, -1] = tile_MRR_compute_latency
    if M_remain * N_remain * K_remain != 0:
        tile_RRR_compute_latency = tile_compute_latency_remain
        tile_latency[-1, -1, -1] = tile_RRR_compute_latency
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
    for m, n, k in self.generate_tile_loops(
        ceil(self.M / M_tile),
        ceil(self.N / N_tile),
        ceil(self.K / K_tile),    
        simdram_loop_order,
    ):
        if m == 0 and n == 0 and k == 0:
            #load data for first tile
            total_latency += M_N_io_latency + M_K_io_latency + K_N_io_latency
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
            # print(total_latency)
        previous_m = m
        previous_n = n
        previous_k = k

    # compute and write last tile
    total_latency += (
        M_N_io_latency
        + tile_compute_latency
    )
    

    # if previous_k > 0:
    #     total_cycle_count += ceil(l2_tiles[-1, -1, -1].K_reduction_cycle_count)
    print(f"gemm latency: {total_latency}")
    return total_latency



def find_arr_tile(self, pcb_module: Device, arr_mapping, debug=False):
    col_per_arr = pcb_module.compute_module.bank.arr_cols
    row_per_arr = pcb_module.compute_module.bank.arr_rows
    if arr_mapping['C'] == 'M':
        arr_tile_M = col_per_arr
        arr_tile_N = find_closest_divisor(row_per_arr // (self.data_type.word_size * 8))
        arr_tile_K = row_per_arr // (self.data_type.word_size * 8) // arr_tile_N
    elif arr_mapping['C'] == 'N':
        arr_tile_N = col_per_arr
        arr_tile_M = find_closest_divisor(row_per_arr // (self.data_type.word_size * 8))
        arr_tile_K = row_per_arr // (self.data_type.word_size * 8) // arr_tile_M
    elif arr_mapping['C'] == 'K':
        arr_tile_K = col_per_arr
        arr_tile_M = find_closest_divisor(row_per_arr // (self.data_type.word_size * 8))
        arr_tile_N = row_per_arr // (self.data_type.word_size * 8) // arr_tile_M
    elif arr_mapping['C'] == 'MN' or arr_mapping['C'] == 'NM':
        arr_tile_M = find_closest_divisor(col_per_arr)
        arr_tile_N = col_per_arr // arr_tile_M
        arr_tile_K = row_per_arr // (self.data_type.word_size * 8) // 2
    elif arr_mapping['C'] == 'MK' or arr_mapping['C'] == 'KM':
        arr_tile_M = find_closest_divisor(col_per_arr)
        arr_tile_K = col_per_arr // arr_tile_M
        arr_tile_N = row_per_arr // (self.data_type.word_size * 8) // 2
    elif arr_mapping['C'] == 'NK' or arr_mapping['C'] == 'KN':
        arr_tile_N = find_closest_divisor(col_per_arr)
        arr_tile_K = col_per_arr // arr_tile_N
        arr_tile_M = row_per_arr // (self.data_type.word_size * 8) // 2

    # if arr_mapping == 'RKCMN':
    #     arr_tile_N, arr_tile_M = find_tile_N_M(self, col_per_arr)
    #     # find nearest power of 2 of tile_K that satisfy the row limit
    #     arr_tile_K, accum_bits = find_tile_K(self, row_per_arr)
    # elif arr_mapping == 'RMNCK':
    #     arr_tile_K = col_per_arr
    #     arr_tile_M = min(32, row_per_arr // (self.data_type.word_size * 8) )
    #     arr_tile_N = row_per_arr // (self.data_type.word_size * 8) // arr_tile_M
    if debug:
        print(f"find_arr_tile: arr_tile_M: {arr_tile_M}, arr_tile_K: {arr_tile_K}, arr_tile_N: {arr_tile_N}")
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
        if debug:
            print(f"get_arr_tile_latency: arr_tile_M={arr_tile_M}, arr_tile_N={arr_tile_N}, arr_tile_K={arr_tile_K}, arr_mapping={arr_mapping}, latency={arr_latency}=({arr_tile_M} * {mul_reduce_op_latency} + {acc_op_latency}) * 1e-9, parallelism_utilization={arr_tile_K/col_per_arr}, capacity_utilization={arr_tile_M*arr_tile_N/row_per_arr}")
    self.stats.simd_utilization = max(self.stats.simd_utilization, simd_utilization)
    self.stats.capacity_utilization = max(self.stats.capacity_utilization, capacity_utilization)
    if debug:
        print(f"simd utilization: {simd_utilization}, capacity utilization: {capacity_utilization}")
    return arr_latency

    # if arr_mapping == 'RKCMN':
    #     #per array latency
    #     add_latency_per_array = arr_tile_K * add_op_latency
    #     mul_latency_per_array = arr_tile_K * mul_op_latency
    #     # Add extra accumulation latency
    #     tile_compute_latency = (add_latency_per_array + mul_latency_per_array + acc_op_latency)*1e-9 #acc_op_latency to add with partial sum
    #     print(f"get_arr_tile_latency: arr_tile_M={arr_tile_M}, arr_tile_N={arr_tile_N}, arr_tile_K={arr_tile_K}, arr_mapping={arr_mapping}, latency={tile_compute_latency}, parallelism_utilization={arr_tile_M*arr_tile_N/col_per_arr}, capacity_utilization={arr_tile_K/row_per_arr}")
    # elif arr_mapping == 'RMNCK':
    #     #per array latency
    #     mul_reduce_latency_per_array = arr_tile_M * arr_tile_N * mul_reduce_latency
    #     # Add extra accumulation latency
    #     # assume MN partial sum stores in 1 row instead of 1 column after all the column reductions, therefore only 1 acc_op_latency is added
    #     tile_compute_latency = (mul_reduce_latency_per_array + acc_op_latency)*1e-9
    #     print(f"get_arr_tile_latency: arr_tile_M={arr_tile_M}, arr_tile_N={arr_tile_N}, arr_tile_K={arr_tile_K}, arr_mapping={arr_mapping}, latency={tile_compute_latency}, parallelism_utilization={arr_tile_K/col_per_arr}, capacity_utilization={arr_tile_M*arr_tile_N/row_per_arr}")
    # return tile_compute_latency

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
    if debug:
        print(f"find_tile_size: {tile_size}")
    return tile_size
        # else:
        #     # tiling[key] == None
        #     to_return[key] = 1
    # if tiling == "MANBKD":
    #     #heuristic tiling M to array, N to bank, K to device
    #     M_tile = arr_tile_M * num_array
    #     N_tile = arr_tile_N * num_bank
    #     K_tile = arr_tile_K * num_device
    # elif tiling == "MANBDK":
    #     M_tile = arr_tile_M * num_array
    #     N_tile = arr_tile_N * num_bank * num_device
    #     K_tile = arr_tile_K
    # M_tile = min(M_tile, self.M)
    # K_tile = min(K_tile, self.K)
    # N_tile = min(N_tile, self.N)
   
    # print(f"M_tile:{M_tile}, K_tile:{K_tile}, N_tile:{N_tile}")
    # return M_tile, N_tile, K_tile

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

    if debug:
        print(f"get_tile_latency: tile_size: {tile_size}, arr_tile_size: {arr_tile_size}, M_K_io_latency: {M_K_io_latency}, K_N_io_latency: {K_N_io_latency}, M_N_io_latency: {M_N_io_latency}, tile_compute_latency:{tile_compute_latency} = {arr_latency}(arr_latency) + {K_reduction_latency}(K_reduction_latency)")
    return (M_K_io_latency, K_N_io_latency, M_N_io_latency, tile_compute_latency)

    # K_N_io_latency = tile_size['K'] * tile_size['N'] * self.data_type.word_size / pcb_module.compute_module.bandwidth #move K_tilexN_tile to compute-dram, consider duplication across arrays
    #heuristic tiling M to array, N to bank, K to device
    # if tiling == "MANBKD":
    #     arr_tile_M = ceil(M_tile / num_array)
    #     arr_tile_N = ceil(N_tile / num_bank)
    #     arr_tile_K = ceil(K_tile / num_device)
        
    #     if broadcast_across_array:
    #         K_N_io_latency = K_tile * N_tile * self.data_type.word_size / pcb_module.compute_module.bandwidth #move K_tilexN_tile to compute-dram, consider broadcasted duplication across arrays
    #     else:
    #         K_N_io_latency = K_tile * N_tile * num_array * self.data_type.word_size / pcb_module.compute_module.bandwidth
    #     if broadcast_across_bank:
    #         M_K_io_latency = M_tile * K_tile * self.data_type.word_size / pcb_module.compute_module.bandwidth #move K_tilexK_tile to compute-dram, consider broadcasted duplication across banks
    #     else:
    #         M_K_io_latency = M_tile * K_tile * num_bank * self.data_type.word_size  / pcb_module.compute_module.bandwidth #move K_tilexK_tile to compute-dram, consider duplication across banks

    #     M_N_io_latency = M_tile * N_tile * self.data_type.word_size / pcb_module.compute_module.bandwidth #move M_tilexN_tile to compute-dram, no duplication as K is reduction axis
    #     K_reduction_latency =  M_tile * N_tile * num_device * self.data_type.word_size / pcb_module.compute_module.bandwidth #each device contains M_tike*N_tile partial sum data, load all these to host and reduce
    #     tile_compute_latency = get_arr_tile_latency(self, pcb_module, arr_tile_M, arr_tile_N, arr_tile_K, arr_mapping) + K_reduction_latency
    # elif tiling == "MANBDK":
    #     arr_tile_M = ceil(M_tile / num_array)
    #     arr_tile_N = ceil(N_tile / num_bank / num_device)
    #     arr_tile_K = ceil(K_tile)

    #     if broadcast_across_array:
    #         K_N_io_latency = K_tile * N_tile * self.data_type.word_size / pcb_module.compute_module.bandwidth #move K_tilexN_tile to compute-dram, consider broadcasted duplication across arrays
    #     else:
    #         K_N_io_latency = K_tile * N_tile * num_array * self.data_type.word_size / pcb_module.compute_module.bandwidth
    #     if broadcast_across_bank:
    #         M_K_io_latency = M_tile * K_tile * num_device *self.data_type.word_size / pcb_module.compute_module.bandwidth #move K_tilexK_tile to compute-dram, consider broadcasted duplication across banks, no broadcast across banks
    #     else:
    #         M_K_io_latency = M_tile * K_tile * num_bank * num_device * self.data_type.word_size  / pcb_module.compute_module.bandwidth #move K_tilexK_tile to compute-dram, consider duplication across banks

    #     M_N_io_latency = M_tile * N_tile * self.data_type.word_size / pcb_module.compute_module.bandwidth #move M_tilexN_tile to compute-dram, no duplication 
    #     K_reduction_latency =  0 # no reduction across device/bank/array on K dimension
    #     tile_compute_latency = get_arr_tile_latency(self, pcb_module, arr_tile_M, arr_tile_N, arr_tile_K, arr_mapping) + K_reduction_latency

    # print(f"M_tile: {M_tile}, K_tile: {K_tile}, N_tile: {N_tile}, M_K_io_latency: {M_K_io_latency}, K_N_io_latency: {K_N_io_latency}, M_N_io_latency: {M_N_io_latency}, tile_compute_latency:{tile_compute_latency}")
    

    # return (M_K_io_latency, K_N_io_latency, M_N_io_latency, tile_compute_latency)

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
        print(f"  matmul: {self.input1_shape}, {self.input2_shape}, {self.output_shape}")
        print(f"  M:{self.M}, K:{self.K}, N:{self.N}")

    self.stats = Stats(strategy)
    # num_col_per_array = pcb_module.compute_module.bank.arr_cols
    # num_row = pcb_module.compute_module.bank.arr_rows
    # num_array = pcb_module.compute_module.bank.arr_count
    # num_bank = pcb_module.compute_module.bank_count
    # num_device = pcb_module.compute_module.bank.device_count
    # num_rank = 1
    # capacity_per_array = num_col_per_array * num_row
    # capacity_per_bank = capacity_per_array * num_array
    # capacity_per_device = capacity_per_bank * num_bank
    # capacity_per_rank = capacity_per_device * num_device
    # total_capacity = capacity_per_rank * num_rank

    ################ Heuristic Tiling #################
    tiling = strategy.tiling
    arr_mapping = strategy.arr_mapping
    loop_order = strategy.loop_order
    broadcast = strategy.broadcast
    if debug:
        print(f"Strategy: tiling: {tiling}, arr_mapping: {arr_mapping}, loop_order: {loop_order}, broadcast: {broadcast}")
    arr_tile_M, arr_tile_N, arr_tile_K = find_arr_tile(self, pcb_module, arr_mapping, debug)
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
        tile_MK_io_latency[:M_t, -1, :K_t], tile_KN_io_latency[:M_t, -1, :K_t], tile_MN_io_latency[:M_t, -1, :K_t], tile_latency[:M_t, -1, :K_t] = get_tile_latency(self, pcb_module, broadcast, tiling, arr_mapping, {'M': M_tile, 'K': K_tile, 'N': N_remain}, debug)
        tile_shape_M[:M_t, -1, :K_t] = M_tile
        tile_shape_N[:M_t, -1, :K_t] = N_remain
        tile_shape_K[:M_t, -1, :K_t] = K_tile
    if K_remain != 0:
        tile_MK_io_latency[:M_t, :N_t, -1], tile_KN_io_latency[:M_t, :N_t, -1], tile_MN_io_latency[:M_t, :N_t, -1], tile_latency[:M_t, :N_t, -1] = get_tile_latency(self, pcb_module, broadcast, tiling, arr_mapping, {'M': M_tile, 'K': K_remain, 'N': N_tile}, debug)
        tile_shape_M[:M_t, :N_t, -1] = M_tile
        tile_shape_N[:M_t, :N_t, -1] = N_tile
        tile_shape_K[:M_t, :N_t, -1] = K_remain
    if M_remain * N_remain != 0:
        tile_MK_io_latency[-1, -1, :K_t], tile_KN_io_latency[-1, -1, :K_t], tile_MN_io_latency[-1, -1, :K_t], tile_latency[-1, -1, :K_t] = get_tile_latency(self, pcb_module, broadcast, tiling, arr_mapping, {'M': M_remain, 'K': K_tile, 'N': N_remain}, debug)
        tile_shape_M[-1, -1, :K_t] = M_remain
        tile_shape_N[-1, -1, :K_t] = N_remain
        tile_shape_K[-1, -1, :K_t] = K_tile
    if M_remain * K_remain != 0:
        tile_MK_io_latency[-1, :N_t, -1], tile_KN_io_latency[-1, :N_t, -1], tile_MN_io_latency[-1, :N_t, -1], tile_latency[-1, :N_t, -1] = get_tile_latency(self, pcb_module, broadcast, tiling, arr_mapping, {'M': M_remain, 'K': K_remain, 'N': N_tile}, debug)
        tile_shape_M[-1, :N_t, -1] = M_remain
        tile_shape_N[-1, :N_t, -1] = N_tile
        tile_shape_K[-1, :N_t, -1] = K_remain
    if N_remain * K_remain != 0:
        tile_MK_io_latency[:M_t, -1, -1], tile_KN_io_latency[:M_t, -1, -1], tile_MN_io_latency[:M_t, -1, -1], tile_latency[:M_t, -1, -1] = get_tile_latency(self, pcb_module, broadcast, tiling, arr_mapping, {'M': M_tile, 'K': K_remain, 'N': N_remain}, debug)
        tile_shape_M[:M_t, -1, -1] = M_tile
        tile_shape_N[:M_t, -1, -1] = N_remain
        tile_shape_K[:M_t, -1, -1] = K_remain
    if M_remain * N_remain * K_remain != 0:
        tile_MK_io_latency[-1, -1, -1], tile_KN_io_latency[-1, -1, -1], tile_MN_io_latency[-1, -1, -1], tile_latency[-1, -1, -1] = get_tile_latency(self, pcb_module, broadcast, tiling, arr_mapping, {'M': M_remain, 'K': K_remain, 'N': N_remain}, debug)
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
    if debug:
        print(f"gemm latency: {total_latency}, compute_latency: {total_compute_latency}, io_latency:{total_io_latency}")
    self.stats.latency = total_latency
    self.stats.compute_latency = total_compute_latency
    self.stats.io_latency = total_io_latency
    return total_latency

def compile_and_simulate_simdram(
    self,
    pcb_module: Device,
    strategy: TilingStrategy,
    debug: bool,
    compile_mode: str = "exhaustive"
):
    
    tiling = strategy.tiling
    arr_mapping = strategy.arr_mapping
    loop_order = strategy.loop_order
    broadcast = strategy.broadcast

    # debug = False
    assert pcb_module.type == "simdram"
    M = self.computational_graph.M
    N = self.computational_graph.N
    K = self.computational_graph.K
    if compile_mode == "heuristic-SIMDRAM":
        return heuristic_simdram(self, pcb_module)     
        
    elif compile_mode == "heuristic-SIMDRAM-broadcast":
        return heuristic_simdram_broadcast(self, pcb_module = pcb_module, strategy=strategy, debug=debug)
       
    elif compile_mode == "heuristic-SIMDRAM-Max":
        return simdram_heuristic_tiling_v2(self, pcb_module,  strategy, debug=debug)
    else:
        raise ValueError(f"compile_mode {compile_mode} not supported")
    
def compile_and_simulate(
    self,
    pcb_module: Device,
    compile_mode: str = "exhaustive",
    debug: bool = False,
):
    assert pcb_module.type == 'simdram'
    return compile_and_simulate_simdram(self,pcb_module, compile_mode, debug)