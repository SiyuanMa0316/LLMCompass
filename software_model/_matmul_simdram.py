import array
from hardware_model.device import Device
from math import ceil, log2, floor
from software_model.utils import Tensor, DataType, simdram_op_latency_dict, simdram_PE_op_latency_dict
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

def simdram_gemv(self, pcb_module: Device, debug = False) -> float:
    
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
    pcb_module.io_module.bandwidth = 19.2 * 8 * (1024/1000) ** 3 # bandwidth in bits per ns
    total_arrays = array_per_device_bank * bank * device * rank

    # case1 M = 1, LHS is a vector of (1 x K), RHS is a complete matrix of (K x N)
    if M == 1:
        tile_N = col_per_array
        tile_M = 1
    
    # M = factor_M * tile_M + remain_M
    # N = factor_N * tile_N + remain_N
    # K = factor_K * tile_K + remain_K

    tile_K, accum_bits = find_tile_K(self, row)
    remain_K = K % tile_K
    factor_K = K // tile_K

    remain_N = N % tile_N
    factor_N = N // tile_N

    factor_M = M // tile_M
    remain_M = M % tile_M

    # total number of complete SIMDRAM launch
    complete_iterations =(factor_K * factor_N * factor_M) / total_arrays


    mac_latency = simdram_op_latency_dict[self.data_type.name]['add'] + simdram_op_latency_dict[self.data_type.name]['mul']
    major_compute_latency = complete_iterations * tile_K * mac_latency 
    
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
    # As each array can hold (1 x tile_K) x (tile_K x tile_N), and remain_K < tile_K and remain_N < tile_N
    # so we only need most factor_N + factor_K + 1 arrays for remaining computation at most
    arrays_for_remaining_tiles = factor_N + factor_K + 1
    remain_iterations = arrays_for_remaining_tiles / total_arrays
    # N dimension: factor_N x [(1, remain_K) x (remain_K, tile_K)] GEMV operations.
    remain_N_compute_latency = tile_K * mac_latency 
    # K dimension: factor_K x [(1, tile_K) x (tile_K, remain_N)] GEMV operations.
    remain_K_compute_latency = remain_K * mac_latency
    overlapping_gemv_compute_latency = remain_K * mac_latency

    # as all remaining workloads are guaranteed to be fulfilled into whole SIMDRAM
    # the dominant latency will be determined by the max(tile_K, remain_K)   
    remain_compute_latency =  max(remain_N_compute_latency, remain_K_compute_latency)

    total_iteration = ceil(complete_iterations + remain_iterations) + 1
    
    # the total iteration determines can we map both remaining and major parts into whole SIMDRAM
    # to overlap the compute latency
    total_compute_latency = total_iteration * mac_latency * tile_K + simdram_op_latency_dict['fp32']['add']

    if debug:
        print(f"SIMDRAM GEMV V2\nM = {M} = {tile_M} * {factor_M} + {remain_M}\nK = {K} = {tile_K} * {factor_K} + {remain_K}\nN = {N} = {tile_N} * {factor_N} + {remain_N}")
        print(f"Bank={bank} Device={device} Array={array_per_device_bank} Total Arrays = {total_arrays} complete_iterations = {complete_iterations}")
        print(f"Complete Iteration: {complete_iterations}, Remaining Iterations: {remain_iterations}, Total Iterations {total_iteration}")
        print(f"Major Compute Latency {major_compute_latency * 1e-6}ms, Remain compute latency {remain_compute_latency * 1e-6}ms")

    total_latency = total_compute_latency
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
    pcb_module.io_module.bandwidth = 19.2 * 8 * (1024/1000) ** 3 # bandwidth in bits per ns
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

    tile_N = col_per_array
    factor_N = floor(N / tile_N)
    remain_N = N % tile_N

    tile_K, accum_bits = find_tile_K(self, row)
    factor_K= floor(K / tile_K)
    remain_K = K % tile_K
    
    # As we are duplicating every tile of LHS vector to exact array of every bank and device
    # and we have broadcasting hardware across bank level, we dont need duplication across bank-level.
    # the M_K_IO_latency = factor_K * tile_K * data_width * device/ BW
    # Meanwhile, we dont need to duplication RHS,
    # so, K_N_IO_latency = factor_K * factor_N * tile_K * tile_N * data_width / BW
    major_M_K_IO_latency = factor_K * tile_K * self.data_type.word_size * device/ pcb_module.io_module.bandwidth
    major_K_N_IO_latency = factor_K * factor_N * tile_K * tile_N * self.data_type.word_size / pcb_module.io_module.bandwidth
    mac_latency = tile_K * (simdram_op_latency_dict[self.data_type.name]['mul'] + simdram_op_latency_dict[self.data_type.name]['add'])
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
        print(f"Partition K -> array: K_iteration {iteration_K}, partition N -> bank and device: N_iteration {iteration_N}")
        print(f"Array required for major tiles :{arrays_for_major_tiles}, Array required for remaining tiles: {arrays_for_remaining_tiles},Total Arrays: {total_arrays}")
        print(f"Major tiles IO latency = {major_IO_latency * 1e-6}ms, Remain tiles IO latency = {remain_IO_latency * 1e-6}ms")


    total_latency = 0
    if total_arrays > (arrays_for_major_tiles + arrays_for_remaining_tiles):
        total_latency = mac_latency + major_IO_latency + remain_IO_latency
    else:
        # cannot schedule remain tiles along with major tiles together
        print(f"Input GEMV size is too large, cannot schedule remain tiles with major tiles together.")

    return total_latency
    

def simdram_heuristic_tiling_v2(self, pcb_module: Device, debug = False) -> float:

    pcb_module.io_module.bandwidth = 19.2 * (1024/1000) ** 3 # bandwidth in bytes per ns
    M = self.computational_graph.M
    N = self.computational_graph.N
    K = self.computational_graph.K

    if M == 1 or N == 1:
        return simdram_gemv_broadcast_only(self,pcb_module, debug)

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
    tile_N, tile_M = self.find_tile_N_M(col_per_array)
    factor_N = ceil(N / tile_N)
    factor_M = ceil(M / tile_M)
    # find nearest power of 2 of tile_K that satisfy the row limit
    tile_K, accum_bits = self.find_tile_K(row)
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
    compute_latency_per_array = add_latency_per_array + mul_latency_per_array + simdram_op_latency_dict['fp32']['add'] + simdram_op_latency_dict['fp32']['mul']
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

def find_arr_tile(self, pcb_module: Device, arr_mapping):
    col_per_arr = pcb_module.compute_module.bank.arr_cols
    row_per_arr = pcb_module.compute_module.bank.arr_rows
    if arr_mapping == 'RKCMN':
        arr_tile_N, arr_tile_M = find_tile_N_M(self, col_per_arr)
        # find nearest power of 2 of tile_K that satisfy the row limit
        arr_tile_K, accum_bits = find_tile_K(self, row_per_arr)
    elif arr_mapping == 'RMNCK':
        arr_tile_K = col_per_arr
        arr_tile_M = min(32, row_per_arr // (self.data_type.word_size * 8) )
        arr_tile_N = row_per_arr // (self.data_type.word_size * 8) // arr_tile_M
    print(f"arr_tile_M: {arr_tile_M}, arr_tile_K: {arr_tile_K}, arr_tile_N: {arr_tile_N}")
    return arr_tile_M, arr_tile_N, arr_tile_K

def get_arr_tile_latency(self, pcb_module: Device, arr_tile_M, arr_tile_N, arr_tile_K, arr_mapping):
    ################ Compute Latencies #################
    #add and mul latency
    if pcb_module.compute_module.with_PE:
        add_op_latency = simdram_PE_op_latency_dict[self.data_type.name]['add']
        mul_op_latency = simdram_PE_op_latency_dict[self.data_type.name]['mul']
        acc_op_latency = simdram_PE_op_latency_dict['int32']['add']
        add_reduce_latency = simdram_PE_op_latency_dict[self.data_type.name]['add_reduce']
    else:
        add_op_latency = simdram_op_latency_dict[self.data_type.name]['add']
        mul_op_latency = simdram_op_latency_dict[self.data_type.name]['mul']
        acc_op_latency = simdram_op_latency_dict['int32']['add']
        add_reduce_latency = simdram_op_latency_dict[self.data_type.name]['add_reduce']
    

    if arr_mapping == 'RKCMN':
        #per array latency
        add_latency_per_array = arr_tile_K * add_op_latency
        mul_latency_per_array = arr_tile_K * mul_op_latency
        # Add extra accumulation latency
        tile_compute_latency = (add_latency_per_array + mul_latency_per_array + acc_op_latency)*1e-9
    elif arr_mapping == 'RMNCK':
        #per array latency
        add_reduce_latency_per_array = arr_tile_M * arr_tile_N * add_reduce_latency
        # Add extra accumulation latency
        # assume MN partial sum stores in 1 row instead of 1 column after all the column reductions, therefore only 1 acc_op_latency is added
        tile_compute_latency = (add_reduce_latency_per_array + acc_op_latency)*1e-9
    return tile_compute_latency

def find_tile_size(self, pcb_module: Device, tiling, arr_tile_M, arr_tile_N, arr_tile_K):
    num_array = pcb_module.compute_module.bank.arr_count
    num_bank = pcb_module.compute_module.bank_count
    num_device = pcb_module.compute_module.bank.device_count
    num_rank = 1
    if tiling == "MANBKD":
        #heuristic tiling M to array, N to bank, K to device
        M_tile = arr_tile_M * num_array
        N_tile = arr_tile_N * num_bank
        K_tile = arr_tile_K * num_device
    elif tiling == "MANBDK":
        M_tile = arr_tile_M * num_array
        N_tile = arr_tile_N * num_bank * num_device
        K_tile = arr_tile_K
    print(f"M_tile:{M_tile}, K_tile:{K_tile}, N_tile:{N_tile}")
    return M_tile, N_tile, K_tile

def get_tile_latency(self, pcb_module: Device, broadcast, tiling, arr_mapping, M_tile, N_tile, K_tile):
    num_array = pcb_module.compute_module.bank.arr_count
    num_bank = pcb_module.compute_module.bank_count
    num_device = pcb_module.compute_module.bank.device_count
    num_rank = 1
    broadcast_across_array = False
    broadcast_across_bank = False
    if "A" in broadcast:
        broadcast_across_array = True
    if "B" in broadcast:
        broadcast_across_bank = True
    #heuristic tiling M to array, N to bank, K to device
    if tiling == "MANBKD":
        arr_tile_M = ceil(M_tile / num_array)
        arr_tile_N = ceil(N_tile / num_bank)
        arr_tile_K = ceil(K_tile / num_device)
        
        if broadcast_across_array:
            K_N_io_latency = K_tile * N_tile * self.data_type.word_size / pcb_module.compute_module.bandwidth #move K_tilexN_tile to compute-dram, consider broadcasted duplication across arrays
        else:
            K_N_io_latency = K_tile * N_tile * num_array * self.data_type.word_size / pcb_module.compute_module.bandwidth
        if broadcast_across_bank:
            M_K_io_latency = M_tile * K_tile * self.data_type.word_size / pcb_module.compute_module.bandwidth #move K_tilexK_tile to compute-dram, consider broadcasted duplication across banks
        else:
            M_K_io_latency = M_tile * K_tile * num_bank * self.data_type.word_size  / pcb_module.compute_module.bandwidth #move K_tilexK_tile to compute-dram, consider duplication across banks

        M_N_io_latency = M_tile * N_tile * self.data_type.word_size / pcb_module.compute_module.bandwidth #move M_tilexN_tile to compute-dram, no duplication as K is reduction axis
        K_reduction_latency =  M_tile * N_tile * num_device * self.data_type.word_size / pcb_module.compute_module.bandwidth #each device contains M_tike*N_tile partial sum data, load all these to host and reduce
        tile_compute_latency = get_arr_tile_latency(self, pcb_module, arr_tile_M, arr_tile_N, arr_tile_K, arr_mapping) + K_reduction_latency
    elif tiling == "MANBDK":
        arr_tile_M = ceil(M_tile / num_array)
        arr_tile_N = ceil(N_tile / num_bank / num_device)
        arr_tile_K = ceil(K_tile)

        if broadcast_across_array:
            K_N_io_latency = K_tile * N_tile * self.data_type.word_size / pcb_module.compute_module.bandwidth #move K_tilexN_tile to compute-dram, consider broadcasted duplication across arrays
        else:
            K_N_io_latency = K_tile * N_tile * num_array * self.data_type.word_size / pcb_module.compute_module.bandwidth
        if broadcast_across_bank:
            M_K_io_latency = M_tile * K_tile * num_device *self.data_type.word_size / pcb_module.compute_module.bandwidth #move K_tilexK_tile to compute-dram, consider broadcasted duplication across banks, no broadcast across banks
        else:
            M_K_io_latency = M_tile * K_tile * num_bank * num_device * self.data_type.word_size  / pcb_module.compute_module.bandwidth #move K_tilexK_tile to compute-dram, consider duplication across banks

        M_N_io_latency = M_tile * N_tile * self.data_type.word_size / pcb_module.compute_module.bandwidth #move M_tilexN_tile to compute-dram, no duplication 
        K_reduction_latency =  0 # no reduction across device/bank/array on K dimension
        tile_compute_latency = get_arr_tile_latency(self, pcb_module, arr_tile_M, arr_tile_N, arr_tile_K, arr_mapping) + K_reduction_latency

    print(f"M_tile: {M_tile}, K_tile: {K_tile}, N_tile: {N_tile}, M_K_io_latency: {M_K_io_latency}, K_N_io_latency: {K_N_io_latency}, M_N_io_latency: {M_N_io_latency}, tile_compute_latency:{tile_compute_latency}")
    

    return (M_K_io_latency, K_N_io_latency, M_N_io_latency, tile_compute_latency)

def heuristic_simdram_broadcast (self, pcb_module: Device, tiling, arr_mapping, loop_order, broadcast):
    '''
    M->array
    N->bank
    K->device
    arr_tile_M * arr_tile_N = arr_cols
    arr_tile_K fit into arr_rows
    loop order: NKM
    '''
    print(f"  matmul: {self.input1_shape}, {self.input2_shape}, {self.output_shape}")
    print(f"  M:{self.M}, K:{self.K}, N:{self.N}")

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
    print(f"tiling: {tiling}, arr_mapping: {arr_mapping}, loop_order: {loop_order}, broadcast: {broadcast}")
    arr_tile_M, arr_tile_N, arr_tile_K = find_arr_tile(self, pcb_module, arr_mapping)
    M_tile, N_tile, K_tile = find_tile_size(self, pcb_module, tiling , arr_tile_M, arr_tile_N, arr_tile_K)
    
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
        tile_MK_io_latency[:M_t, :N_t, :K_t], tile_KN_io_latency[:M_t, :N_t, :K_t], tile_MN_io_latency[:M_t, :N_t, :K_t], tile_latency[:M_t, :N_t, :K_t] = get_tile_latency(self, pcb_module, broadcast, tiling, arr_mapping, M_tile, N_tile, K_tile)
        tile_shape_M[:M_t, :N_t, :K_t] = M_tile
        tile_shape_N[:M_t, :N_t, :K_t] = N_tile
        tile_shape_K[:M_t, :N_t, :K_t] = K_tile
    if M_remain != 0:
        tile_MK_io_latency[-1, :N_t, :K_t], tile_KN_io_latency[-1, :N_t, :K_t], tile_MN_io_latency[-1, :N_t, :K_t], tile_latency[-1, :N_t, :K_t] = get_tile_latency(self, pcb_module, broadcast, tiling, arr_mapping, M_remain, N_tile, K_tile)
        tile_shape_M[-1, :N_t, :K_t] = M_remain
        tile_shape_N[-1, :N_t, :K_t] = N_tile
        tile_shape_K[-1, :N_t, :K_t] = K_tile
    if N_remain != 0:
        tile_MK_io_latency[:M_t, -1, :K_t], tile_KN_io_latency[:M_t, -1, :K_t], tile_MN_io_latency[:M_t, -1, :K_t], tile_latency[:M_t, -1, :K_t] = get_tile_latency(self, pcb_module, broadcast, tiling, arr_mapping, M_tile, N_remain, K_tile)
        tile_shape_M[:M_t, -1, :K_t] = M_tile
        tile_shape_N[:M_t, -1, :K_t] = N_remain
        tile_shape_K[:M_t, -1, :K_t] = K_tile
    if K_remain != 0:
        tile_MK_io_latency[:M_t, :N_t, -1], tile_KN_io_latency[:M_t, :N_t, -1], tile_MN_io_latency[:M_t, :N_t, -1], tile_latency[:M_t, :N_t, -1] = get_tile_latency(self, pcb_module, broadcast, tiling, arr_mapping, M_tile, N_tile, K_remain)
        tile_shape_M[:M_t, :N_t, -1] = M_tile
        tile_shape_N[:M_t, :N_t, -1] = N_tile
        tile_shape_K[:M_t, :N_t, -1] = K_remain
    if M_remain * N_remain != 0:
        tile_MK_io_latency[-1, -1, :K_t], tile_KN_io_latency[-1, -1, :K_t], tile_MN_io_latency[-1, -1, :K_t], tile_latency[-1, -1, :K_t] = get_tile_latency(self, pcb_module, broadcast, tiling, arr_mapping, M_remain, N_remain, K_tile)
        tile_shape_M[-1, -1, :K_t] = M_remain
        tile_shape_N[-1, -1, :K_t] = N_remain
        tile_shape_K[-1, -1, :K_t] = K_tile
    if M_remain * K_remain != 0:
        tile_MK_io_latency[-1, :N_t, -1], tile_KN_io_latency[-1, :N_t, -1], tile_MN_io_latency[-1, :N_t, -1], tile_latency[-1, :N_t, -1] = get_tile_latency(self, pcb_module, broadcast, tiling, arr_mapping, M_remain, N_tile, K_remain)
        tile_shape_M[-1, :N_t, -1] = M_remain
        tile_shape_N[-1, :N_t, -1] = N_tile
        tile_shape_K[-1, :N_t, -1] = K_remain
    if N_remain * K_remain != 0:
        tile_MK_io_latency[:M_t, -1, -1], tile_KN_io_latency[:M_t, -1, -1], tile_MN_io_latency[:M_t, -1, -1], tile_latency[:M_t, -1, -1] = get_tile_latency(self, pcb_module, broadcast, tiling, arr_mapping, M_tile, N_remain, K_remain)
        tile_shape_M[:M_t, -1, -1] = M_tile
        tile_shape_N[:M_t, -1, -1] = N_remain
        tile_shape_K[:M_t, -1, -1] = K_remain
    if M_remain * N_remain * K_remain != 0:
        tile_MK_io_latency[-1, -1, -1], tile_KN_io_latency[-1, -1, -1], tile_MN_io_latency[-1, -1, -1], tile_latency[-1, -1, -1] = get_tile_latency(self, pcb_module, broadcast, tiling, arr_mapping, M_remain, N_remain, K_remain)
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
        tile_MN_io_latency[-1, -1, -1] #last tile write
        + tile_latency[-1, -1, -1] #lsat tile compute
    )
    

    # if previous_k > 0:
    #     total_cycle_count += ceil(l2_tiles[-1, -1, -1].K_reduction_cycle_count)
    print(f"gemm latency: {total_latency}")
    return total_latency

def compile_and_simulate_simdram(
    self,
    pcb_module: Device,
    compile_mode: str = "exhaustive",   
):
    debug = False
    assert pcb_module.type == "simdram"
    M = self.computational_graph.M
    N = self.computational_graph.N
    K = self.computational_graph.K
    if compile_mode == "heuristic-SIMDRAM":
        return heuristic_simdram(self, pcb_module)     
        
    elif compile_mode == "heuristic-SIMDRAM-broadcast":
        return heuristic_simdram_broadcast(self, pcb_module = pcb_module, tiling="MANBKD", arr_mapping="RKCMN", loop_order="nkm", broadcast="")
       
    elif compile_mode == "heuristic-SIMDRAM-Max":
        return simdram_heuristic_tiling_v2(self, pcb_module, True)
    else:
        raise ValueError(f"compile_mode {compile_mode} not supported")
    
def compile_and_simulate(
    self,
    pcb_module: Device,
    compile_mode: str = "exhaustive",
    debug: bool = False,
):
    assert pcb_module.type == 'simdram'
    return compile_and_simulate_simdram(self,pcb_module, compile_mode)