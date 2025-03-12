from hardware_model.device import Device
from math import ceil, log2, floor
from software_model.utils import Tensor, DataType, simdram_op_latency_dict
import numpy as np

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
    
    """Wrapper function to tiling and estimate GEMV workload latency

    Args:
        pcb_module (Device): input device

    Returns:
        float: estimated total latency for input vector
    """
    M = self.computational_graph.M
    N = self.computational_graph.N
    K = self.computational_graph.K
    assert N == 1 or M == 1, f"SIMDRAM_GEMV require input M = 1 or N = 1"
    col_per_array = 128
    row = 131072
    array_per_device_bank = 64
    bank = 16
    device = 8
    rank = 1
    pcb_module.io_module.bandwidth = 19.2 * 8 * (1024/1000) ** 3 # bandwidth in bits per ns
    total_arrays = array_per_device_bank * bank * device * rank
    """
    the smallest workload size is 1 x tile_K x tile_N, and tile_N = col
    Assume LHS is A:        A(1,1)     A(1,2)     A(1,3) ...    A(1, factor_K)
    Assume RHS is B:        B(1,1)     B(1,2)     B(1,3) ...    B(1, factor_N)
                            B(2,1)     B(2,2)     B(2,3) ...    B(2, factor_N)
                            ...         
                            B(factor_K,1)       ...             B(factor_K, factor_N)
    We are mapping every tile of A to and every tile of the the column vector of B into every array.
    M = 1, so tile_M = 1, tile_N = col_per_array. 
    N = factor_N * tile_N + remain_N.
    K = factor_K * tile_K + remain_K.
    
    Overall, we processes factor_N x factor_K x (1 x tile_K x tile_N) small GEMVs.
    we are mapping total 'factor_N x factor_K' tiles into all Array(A) x Bank(B) x Device(D)) SIMDRAM arrays.
    
    Theoretically we need 'factor_N x factor_K / (A * B * D)' iterations of fulfilled SIMDRAM execution.
    In these iterations: SIMDRAM is fully utilized for full_iteration = 'floor(factor_N x factor_K / (A * B * D))' iterations
    each will have exactly overlapped latency = tile_K * T * full_iteration, T = t_add + t_mul.

    Resultant, we have used_array = 'factor_N x factor_K - full_iteration * (A * B * D)' used arrays in the final SIMDRAM iteration.
    and have free_array = A * B * D - used_array to process our remaining tiles.
    In this case, we can schedule a part of the remaining tiles into these free arrays, 
    which will have overlapped latency = tile_K * T since remain_K will always smaller or equal to tile_K.

    Two remaining parts in B: a row vector remain_K x N, and a column vector K x tile_N, requiring
    total (factor_K + factor_N - 1) arrays for computation. We schedule the remaining tiles using free_array and compute the final
    remaining parts. In this case, we still need 'factor_K + factor_N - 1 - free_array' to be used, which will definitely smaller than
    normal A * B * D. So, the remaining latency = remain_K * T.
    Total latency = major_compute_latency =  tile_K * T * full_iteration + tile_K * T + remain_K * T.
    where full_iteration = floor(factor_M * factor_N * floor(K / tile_K) / (A * B *D)) 
    since tile_K is the dominating factor of the overall latency, we can simply create an broadcasted numpy array
    to find the optimal tile_K and get return the latency
    """
    if M == 1:
        tile_M = 1
        tile_N = col_per_array
        factor_N = floor(N / tile_N)
        factor_M = 1
        remain_M_N = N - factor_N * tile_N
    if N == 1:
        tile_N = 1
        tile_M = col_per_array
        factor_M = floor(M / tile_M)
        factor_N = 1
        remain_M_N = M - factor_M * tile_M

    accum_bits = 32
    tile_K_limit = (row - 32) // (2 * self.data_type.word_size * 8) # maximum of tile_K in one array
    tile_K_list = np.arange(1, tile_K_limit)
    factor_K_list = np.floor(K / tile_K_list)
    remain_K_list = K - factor_K_list * tile_K_list
    full_iteration = np.floor(factor_M * factor_N * factor_K_list / (total_arrays))
    op_latency = (simdram_op_latency_dict[self.data_type.name]['add'] + simdram_op_latency_dict[self.data_type.name]['mul'])  
    compute_latency_list = tile_K_list * op_latency * full_iteration + tile_K_list * op_latency + remain_K_list * op_latency
    write_back_latency = M * N * accum_bits * 8 / pcb_module.io_module.bandwidth
    compute_latency_min_idx = np.argmin(compute_latency_list)
    compute_latency = compute_latency_list[compute_latency_min_idx]
    tile_K = tile_K_list[compute_latency_min_idx]
    factor_K = factor_K_list[compute_latency_min_idx]
    remain_K = remain_K_list[compute_latency_min_idx]
    total_latency = compute_latency + write_back_latency
    
    
    if debug:
        print(f"SIMDRAM - GEMV: tile_M {tile_M} tile_N {tile_N} tile_K {tile_K}, factor_M {factor_M}, factor_N {factor_N}, factor_K {factor_K}")
        print(f"remain_K {remain_K}, remain_M_N {remain_M_N}, tile_k_limits {tile_K_limit}")
        print(f"total latency {total_latency} ns")
    return total_latency 

def simdram_heuristic_tiling_v2(self, pcb_module: Device, debug = False) -> float:

    pcb_module.io_module.bandwidth = 19.2 * 8 * (1024/1000) ** 3 # bandwidth in bits per ns
    M = self.computational_graph.M
    N = self.computational_graph.N
    K = self.computational_graph.K

    if M == 1 or N == 1:
        return self.simdram_gemv(pcb_module, debug)

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
        factor_N = ceil(N / arr_tile_N)
        factor_M = ceil(M / arr_tile_M)
        # find nearest power of 2 of tile_K that satisfy the row limit
        arr_tile_K, accum_bits = find_tile_K(self, num_row)
        factor_K = floor(K / arr_tile_K)
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
    
        M_t = M // M_tile
        N_t = N // N_tile
        K_t = K // K_tile
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
            [ceil(M / M_tile), ceil(N / N_tile), ceil(K / K_tile)]
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
            ceil(M / M_tile),
            ceil(N / N_tile),
            ceil(K / K_tile),    
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
    elif compile_mode == "heuristic-SIMDRAM-broadcast":
        
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
        factor_N = ceil(N / arr_tile_N)
        factor_M = ceil(M / arr_tile_M)
        # find nearest power of 2 of tile_K that satisfy the row limit
        arr_tile_K, accum_bits = find_tile_K(self, num_row)
        factor_K = floor(K / arr_tile_K)

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

        K_N_io_latency = K_tile * N_tile * self.data_type.word_size / pcb_module.compute_module.bandwidth #move K_tilexN_tile to compute-dram, consider broadcasted duplication across arrays
        M_K_io_latency = M_tile * K_tile * self.data_type.word_size / pcb_module.compute_module.bandwidth #move K_tilexK_tile to compute-dram, consider broadcasted duplication across banks
        M_N_io_latency = M_tile * N_tile * self.data_type.word_size / pcb_module.compute_module.bandwidth #move M_tilexN_tile to compute-dram, no duplication as K is reduction axis
    
        M_t = M // M_tile
        N_t = N // N_tile
        K_t = K // K_tile
        M_remain = self.M % M_tile
        N_remain = self.N % N_tile
        K_remain = self.K % K_tile
        print(f"M_tile:{M_tile}, K_tile:{K_tile}, N_tile:{N_tile}, M_remain:{M_remain}, K_remain:{K_remain}, N_remain:{N_remain}")
        arr_K_remain = ceil(K_remain / num_device)

        #per array latency
        add_latency_per_array = arr_tile_K * simdram_op_latency_dict[self.data_type.name]['add']
        mul_latency_per_array = arr_tile_K * simdram_op_latency_dict[self.data_type.name]['mul']
        # Add extra accumulation latency
        tile_compute_latency = (add_latency_per_array + mul_latency_per_array + simdram_op_latency_dict['fp32']['add'] + simdram_op_latency_dict['fp32']['mul'])*1e-9
        # reduction across devices
        tile_compute_latency += M_tile * N_tile * num_device * self.data_type.word_size / pcb_module.compute_module.bandwidth # no efficient broadcast across device

        #per array latency for K_remain
        add_latency_per_array_remain = arr_K_remain * simdram_op_latency_dict[self.data_type.name]['add']
        mul_latency_per_array_remain = arr_K_remain * simdram_op_latency_dict[self.data_type.name]['mul']
        # Add extra accumulation latency
        tile_compute_latency_remain = (add_latency_per_array_remain + mul_latency_per_array_remain + simdram_op_latency_dict['fp32']['add'] + simdram_op_latency_dict['fp32']['mul'])*1e-9
        # reduction across devices
        tile_compute_latency_remain += M_remain * N_remain * num_device * self.data_type.word_size / pcb_module.compute_module.bandwidth # no efficient broadcast across device
        
        print(f"K_N_io_latency: {K_N_io_latency}, M_K_io_latency: {M_K_io_latency}, M_N_io_latency: {M_N_io_latency}, tile_compute_latency:{tile_compute_latency}, tile_compute_latency_remain:{tile_compute_latency_remain}")

        tile_latency = np.zeros(
            [ceil(M / M_tile), ceil(N / N_tile), ceil(K / K_tile)]
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
            ceil(M / M_tile),
            ceil(N / N_tile),
            ceil(K / K_tile),    
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
    elif compile_mode == "heuristic-SIMDRAM-Max":
        return self.simdram_heuristic_tiling_v2(pcb_module, True)
    else:
        raise ValueError(f"compile_mode {compile_mode} not supported")
    
def compile_and_simulate(
    self,
    pcb_module: Device,
    compile_mode: str = "exhaustive",
    debug: bool = False,
):
    assert pcb_module.type == 'simdram'
    return self.compile_and_simulate_simdram(pcb_module, compile_mode)