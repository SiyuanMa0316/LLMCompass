import array
from pyparsing import col
from sympy import factor
from utils import size
from typing import List, Tuple
from hardware_model.device import Device
from software_model.operators import Operator
from software_model.utils import Tensor, DataType, simdram_op_latency_dict
from math import ceil, log2, floor
import torch
import time
import statistics
import numpy as np
import pandas as pd
import os
from scalesim.scale_sim import scalesim
import copy
from pimsab_exp.run_pimsab_gemm import run_gemm


class BatchedMatmul(Operator):
    def __init__(self, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.input1_shape = None
        self.input2_shape = None
        self.output_shape = None

    def __call__(self, input1: Tensor, input2: Tensor) -> Tensor:
        # [b, M, K] * [b, K, N] = [b, M, N]
        assert self.data_type == input1.data_type
        assert self.data_type == input2.data_type
        self.input1_shape = input1.shape
        self.input2_shape = input2.shape
        assert size(self.input1_shape[:-2]) == size(self.input2_shape[:-2])
        self.bs = size(self.input1_shape[:-2])
        self.M = self.input1_shape[-2]
        self.K = self.input1_shape[-1]
        assert self.input2_shape[-2] == self.K
        self.N = self.input2_shape[-1]
        self.output_shape = self.input1_shape[:-2] + [self.M, self.N]
        output = Tensor(self.output_shape, self.data_type)
        return output

    def roofline_model(self, pcb_module: Device):
        matmul = Matmul(self.data_type)
        _ = matmul(Tensor([self.M, self.K]), Tensor([self.K, self.N]))
        matmul_latency = matmul.roofline_model(pcb_module)
        self.roofline_latency = matmul_latency * self.bs
        
        self.roofline_latency = max (
            self.flop_count / pcb_module.compute_module.total_systolic_array_flops,
            self.io_count * self.data_type.word_size
            / min(
                pcb_module.io_module.bandwidth,
                pcb_module.compute_module.l2_bandwidth_per_cycle * pcb_module.compute_module.clock_freq,
            ),
        )
        
        return self.roofline_latency

    # def compile_and_simulate(self, pcb_module: Device, compile_mode: str):
    #     matmul = Matmul(self.data_type)
    #     _ = matmul(Tensor([self.M, self.K]), Tensor([self.K, self.N]))
    #     matmul_latency = (
    #         matmul.compile_and_simulate(pcb_module, compile_mode)
    #         # - pcb_module.io_module.latency * 2
    #     )
    #     self.latency = matmul_latency * self.bs  # + pcb_module.io_module.latency * 2
    #     return self.latency

    def compile_and_simulate(self, pcb_module: Device, compile_mode: str):
        matmul = Matmul(self.data_type)
        _ = matmul(Tensor([self.M, self.K],self.data_type), Tensor([self.K, self.N],self.data_type))
        matmul_latency1 = (
            matmul.compile_and_simulate(pcb_module, compile_mode) * self.bs
        )

        matmul = Matmul(self.data_type)
        _ = matmul(
            Tensor([self.M, self.K * self.bs],self.data_type), Tensor([self.K * self.bs, self.N],self.data_type)
        )
        matmul_latency2 = (
            matmul.compile_and_simulate(pcb_module, compile_mode)
            # Siyuan: I don't understand this overhead so commented out
            # + (self.bs - 1)
            # * self.M
            # * self.N
            # * self.data_type.word_size
            # / pcb_module.io_module.bandwidth
        )
        self.latency = min(matmul_latency1, matmul_latency2)
        print(f"BatchedMatmul latency: {self.latency}")
        return self.latency

    def run_on_gpu(
        self,
    ):
        input1 = torch.randn(self.bs, self.M, self.K, dtype=torch.float16).cuda()
        input2 = torch.randn(self.bs, self.K, self.N, dtype=torch.float16).cuda()
        latencies = []
        # warmup
        for _ in range(3):
            _ = torch.bmm(input1, input2)
            torch.cuda.synchronize()
        for _ in range(self.iterations):
            start = time.time()
            output = torch.bmm(input1, input2)
            torch.cuda.synchronize()
            end = time.time()
            latencies.append(end - start)

        self.latency_on_gpu = (
            statistics.median(latencies)
            # - self.gpu_kernel_launch_overhead()
            # - 4e-5
            # min(latencies) - 8e-6
        )  # GPU launch kernel overhead and PyTorch overhead
        return self.latency_on_gpu

    @staticmethod
    def gpu_kernel_launch_overhead():
        latencies = []
        for _ in range(50):
            a = torch.randn(1, 1, 1, device="cuda")
            b = torch.randn(1, 1, 1, device="cuda")
            torch.cuda.synchronize()
            start = time.time()
            c = torch.bmm(a, b)
            torch.cuda.synchronize()
            end = time.time()
            latencies.append(end - start)
        avg_overhead = statistics.median(latencies)
        # print('GPU kernel launch overhead: ', avg_overhead*1e3, 'ms')
        # print(latencies)
        return avg_overhead


class Matmul(Operator):
    def __init__(self, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.input1_shape = None
        self.input2_shape = None
        self.output_shape = None
        self.look_up_table = None
        self.best_mapping = None

    def __call__(self, input1: Tensor, input2: Tensor) -> Tensor:
        # [bs, M, K] * [K, N] = [bs, M, N]
        # print(self.data_type.name)
        # print(input1.data_type.name)
        assert self.data_type == input1.data_type
        assert self.data_type == input2.data_type
        self.input1_shape = input1.shape
        self.input2_shape = input2.shape
        self.M = size(self.input1_shape[:-1])
        self.K = self.input1_shape[-1]
        assert self.input2_shape[-2] == self.K
        self.N = self.input2_shape[-1]
        if len(self.input1_shape) == 2:
            self.output_shape = [self.M, self.N]
        else:
            self.output_shape = self.input1_shape[:-1] + [self.N]
        output = Tensor(self.output_shape, self.data_type)
        self.computational_graph = self.ComputationalGraph(
            self.M, self.N, self.K, self.data_type
        )
        self.flop_count = 2 * self.M * self.K * self.N
        self.io_count = self.M * self.K + self.K * self.N + self.M * self.N
        # print(f'{self.M}, {self.N}, {self.K}')
        return output

    def roofline_model(self, pcb_module: Device):
        if self.M ==1 or self.N == 1:
            vector_flops = (pcb_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
                    * pcb_module.compute_module.core_count
                    * pcb_module.compute_module.clock_freq)
            compute_latency = (
                    self.flop_count
                    / pcb_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
                    / pcb_module.compute_module.core_count
                    / pcb_module.compute_module.clock_freq
                )
        else:
            compute_latency = self.flop_count / pcb_module.compute_module.total_systolic_array_flops
        self.roofline_latency = max(
            compute_latency,
            #Siyuan bug fix:
            self.io_count * self.data_type.word_size
            / min(
                pcb_module.io_module.bandwidth,
                pcb_module.compute_module.l2_bandwidth_per_cycle
                * pcb_module.compute_module.clock_freq,
            ),
        )
        print(f"  flop_count:{self.flop_count}, compute_flops:{pcb_module.compute_module.total_systolic_array_flops},  io_count:{self.io_count}, io_bw:{pcb_module.io_module.bandwidth}, l2_bw:{pcb_module.compute_module.l2_bandwidth_per_cycle * pcb_module.compute_module.clock_freq}, latency: {self.roofline_latency}")
        return self.roofline_latency

    def print_latency(self):
        print(
            f"{self.computational_graph.M}, {self.computational_graph.N}, {self.computational_graph.K}, {self.best_latency*1e3:.4f}ms, {self.latency_on_gpu*1e3:.4f}ms, {self.best_latency/self.latency_on_gpu*100:.2f}%",
            flush=True,
        )

    @staticmethod
    def generate_tile_loops(loop_M: int, loop_N: int, loop_K: int, loop_order: str):
        assert loop_order in ["mkn", "mnk", "nkm", "nmk", "knm", "kmn"]
        if loop_order == "mnk":
            for m in range(loop_M):
                for n in range(loop_N):
                    for k in range(loop_K):
                        yield m, n, k
        elif loop_order == "mkn":
            for m in range(loop_M):
                for k in range(loop_K):
                    for n in range(loop_N):
                        yield m, n, k
        elif loop_order == "nmk":
            for n in range(loop_N):
                for m in range(loop_M):
                    for k in range(loop_K):
                        yield m, n, k
        elif loop_order == "nkm":
            for n in range(loop_N):
                for k in range(loop_K):
                    for m in range(loop_M):
                        yield m, n, k
        elif loop_order == "knm":
            for k in range(loop_K):
                for n in range(loop_N):
                    for m in range(loop_M):
                        yield m, n, k
        elif loop_order == "kmn":
            for k in range(loop_K):
                for m in range(loop_M):
                    for n in range(loop_N):
                        yield m, n, k

    class ComputationalGraph:
        def __init__(self, M: int, N: int, K: int, data_type: DataType):
            self.M = M
            self.N = N
            self.K = K
            self.data_type = data_type

        def display(self):
            print("-" * 10 + " Computational Graph " + "-" * 10)
            print(
                f"M: {self.M}, N: {self.N}, K: {self.K}, word_size(B): {self.data_type.word_size}"
            )

    class Mapping:
        def __init__(
            self,
            l2_tile_M: int,
            l2_tile_N: int,
            l2_tile_K: int,
            is_l2_double_buffering: bool,
            l1_tile_M: int,
            l1_tile_N: int,
            l1_tile_K: int,
            l2_loop_order: str,
            l1_loop_order: str,
            l0_M_tiling_factor: int,
            l0_N_tiling_factor: int,
            l0_K_tiling_factor: int,
            dataflow: str = "os",
        ):
            self.l2_tile_M = l2_tile_M
            self.l2_tile_N = l2_tile_N
            self.l2_tile_K = l2_tile_K
            self.is_l2_double_buffering = is_l2_double_buffering
            self.l1_tile_M = l1_tile_M
            self.l1_tile_N = l1_tile_N
            self.l1_tile_K = l1_tile_K
            self.l2_loop_order = l2_loop_order
            self.l1_loop_order = l1_loop_order
            self.l0_M_tiling_factor = l0_M_tiling_factor
            self.l0_N_tiling_factor = l0_N_tiling_factor
            self.l0_K_tiling_factor = l0_K_tiling_factor
            self.dataflow = dataflow

        def display(self):
            print(f'{"-"*10} Mapping {"-"*10}')
            print(
                f"l2_tile_M: {self.l2_tile_M}, l2_tile_N: {self.l2_tile_N}, l2_tile_K: {self.l2_tile_K}, is_l2_double_buffering: {self.is_l2_double_buffering}, l2_loop_order: {self.l2_loop_order}"
            )
            print(
                f"l1_tile_M: {self.l1_tile_M}, l1_tile_N: {self.l1_tile_N}, l1_tile_K: {self.l1_tile_K}, l1_loop_order: {self.l1_loop_order}"
            )
            print(
                f"l0_M_tiling_factor: {self.l0_M_tiling_factor}, l0_N_tiling_factor: {self.l0_N_tiling_factor}, l0_K_tiling_factor: {self.l0_K_tiling_factor}"
            )

    @staticmethod
    def find_permutations(n):
        permutations = set()

        for i in range(1, n + 1):
            if n % i == 0:
                for j in range(1, n + 1):
                    if (n // i) % j == 0:
                        k = n // (i * j)
                        permutations.add((i, j, k))

        return list(permutations)

 
    
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

    def compile_and_simulate_systolic(
        self,
        pcb_module: Device,
        compile_mode: str = "exhaustive",
    ):
        assert pcb_module.type == 'systolic'
        min_cycle_count = 2**63 - 1
        best_mapping = None
        M = self.computational_graph.M
        N = self.computational_graph.N
        K = self.computational_graph.K
        if (M == 1 or N == 1) and (
            compile_mode == "heuristic-GPU"
            or compile_mode == "heuristic-our-throughput"
        ):
            working_set_size = M * K + N * K + M * N
            total_io_count = working_set_size * self.data_type.word_size
            io_latency = total_io_count / pcb_module.io_module.bandwidth
            total_flop_count = 2 * M * N * K
            compute_latency = (
                total_flop_count
                / pcb_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
                / pcb_module.compute_module.core_count
                / pcb_module.compute_module.clock_freq
            )
            self.latency = max(
                compute_latency, io_latency
            )  # + pcb_module.io_module.latency * 2
            return self.latency
        if compile_mode == "exhaustive":
            for l2_tile_M_log2 in range(5, ceil(log2(self.computational_graph.M)) + 1):
                #partition M dimension by 32, 64, 128, ....
                l2_tile_M = 2**l2_tile_M_log2
                for l2_tile_N_log2 in range(
                    # partition N dimension by 32, 64, 128, ...
                    5, ceil(log2(self.computational_graph.N)) + 1
                ):
                    l2_tile_N = 2**l2_tile_N_log2
                    for l2_tile_K_log2 in range(
                        # partition K dimension by 32, 64, 128 ...
                        5, ceil(log2(self.computational_graph.K)) + 1
                    ):
                        l2_tile_K = 2**l2_tile_K_log2
                        working_set_size = (
                            l2_tile_N * l2_tile_K
                            + l2_tile_M * l2_tile_K
                            + l2_tile_M * l2_tile_N
                        )
                        if (
                            working_set_size
                            # partition group does not fit into L2 cache, ignore current partition
                            > pcb_module.compute_module.l2_size
                            // self.data_type.word_size
                        ):
                            continue
                        elif (
                            # partition group smaller than 1/2 size of L2 cache, double buffering 
                            working_set_size
                            <= pcb_module.compute_module.l2_size
                            // self.data_type.word_size
                            // 2
                        ):
                            is_l2_double_buffering = True
                        else:
                            is_l2_double_buffering = False
                            
                            #sub-partition workgroup into L1 cache, following same philosophy as previous,
                        for l1_tile_M_log2 in range(5, l2_tile_M_log2 + 1):
                            l1_tile_M = 2**l1_tile_M_log2
                            for l1_tile_N_log2 in range(5, l2_tile_N_log2 + 1):
                                l1_tile_N = 2**l1_tile_N_log2
                                for l1_tile_K_log2 in range(5, l2_tile_K_log2 + 1):
                                    l1_tile_K = 2**l1_tile_K_log2
                                    # If worksize > 1/2 L1 cache size
                                    if (
                                        l1_tile_M * l1_tile_N
                                        + l1_tile_N * l1_tile_K
                                        + l1_tile_M * l1_tile_K
                                        > pcb_module.compute_module.core.SRAM_size
                                        // self.data_type.word_size
                                        // 2
                                    ):
                                        continue
                                    for l2_loop_order in [
                                        "mkn",
                                        "mnk",
                                        "nkm",
                                        "nmk",
                                        "knm",
                                        "kmn",
                                    ]:
                                        for l1_loop_order in [
                                            "mkn",
                                            "mnk",
                                            "nkm",
                                            "nmk",
                                            "knm",
                                            "kmn",
                                        ]:
                                            for (
                                                l0_M_tiling_factor,
                                                l0_N_tiling_factor,
                                                l0_K_tiling_factor,
                                            ) in self.find_permutations(
                                                pcb_module.compute_module.core.systolic_array_count
                                            ):
                                                # mapping tiling factor based on systolic array count.
                                                mapping = self.Mapping(
                                                    l2_tile_M,
                                                    l2_tile_N,
                                                    l2_tile_K,
                                                    is_l2_double_buffering,
                                                    l1_tile_M,
                                                    l1_tile_N,
                                                    l1_tile_K,
                                                    l2_loop_order,
                                                    l1_loop_order,
                                                    l0_M_tiling_factor,
                                                    l0_N_tiling_factor,
                                                    l0_K_tiling_factor,
                                                )
                                                cycle_count = self.simulate(
                                                    self.computational_graph,
                                                    mapping,
                                                    pcb_module,
                                                )
                                                if cycle_count < min_cycle_count:
                                                    min_cycle_count = cycle_count
                                                    best_mapping = mapping
        elif compile_mode == "heuristic-our-throughput":
            i = 0
            for l2_tile_M in [32, 64, 128, 256, 512, 1024, 2048, 4096]:
                for l2_tile_N in [
                    l2_tile_M // 4,
                    l2_tile_M // 2,
                    l2_tile_M,
                    l2_tile_M * 2,
                    l2_tile_M * 4,
                    l2_tile_M * 8,
                    l2_tile_M * 16,
                    l2_tile_M * 32,
                    
                ]:
                    l2_tile_K_max = (
                        pcb_module.compute_module.l2_size
                        // self.data_type.word_size
                        // 2
                        - l2_tile_M * l2_tile_N
                    ) // (l2_tile_M + l2_tile_N)
                    if l2_tile_K_max < 1:
                        continue
                    l2_tile_K = min(l2_tile_K_max, K)
                    l2_tile_K = floor(log2(l2_tile_K))
                    l2_tile_K = 2**l2_tile_K
                    working_set_size = (
                        l2_tile_N * l2_tile_K
                        + l2_tile_M * l2_tile_K
                        + l2_tile_M * l2_tile_N
                    )
                    if (
                        working_set_size
                        > pcb_module.compute_module.l2_size // self.data_type.word_size
                    ):
                        continue
                    elif (
                        working_set_size
                        <= pcb_module.compute_module.l2_size
                        // self.data_type.word_size
                        // 2
                    ):
                        is_l2_double_buffering = True
                    else:
                        is_l2_double_buffering = False

                    assert is_l2_double_buffering

                    for l1_tile_M in [32, 64, 128, 256]:
                        l1_tile_M = min(l1_tile_M, l2_tile_M, l2_tile_N)
                        # if l1_tile_M > min(l2_tile_M, l2_tile_N):
                        #     continue
                        l1_tile_N = l1_tile_M
                        l1_tile_K_max = (
                            pcb_module.compute_module.core.SRAM_size
                            // self.data_type.word_size
                            // 2
                            - l1_tile_M * l1_tile_N
                        ) // (l1_tile_M + l1_tile_N)
                        if l1_tile_K_max < 1:
                            continue
                        l1_tile_K = min(l1_tile_K_max, l2_tile_K)
                        l1_tile_K = floor(log2(l1_tile_K))
                        l1_tile_K = 2**l1_tile_K

                        if (
                            l1_tile_M * l1_tile_N
                            + l1_tile_N * l1_tile_K
                            + l1_tile_M * l1_tile_K
                            > pcb_module.compute_module.core.SRAM_size
                            // self.data_type.word_size
                            // 2
                        ):
                            continue
                        l2_loop_order = "knm"
                        l1_loop_order = "knm"
                        for (
                            l0_M_tiling_factor,
                            l0_N_tiling_factor,
                            l0_K_tiling_factor,
                        ) in [(2, 2, 1)]:
                            # self.find_permutations(
                            #     pcb_module.compute_module.core.systolic_array_count
                            # ):
                            i += 1
                            # start = time.time()
                            mapping = self.Mapping(
                                l2_tile_M,
                                l2_tile_N,
                                l2_tile_K,
                                is_l2_double_buffering,
                                l1_tile_M,
                                l1_tile_N,
                                l1_tile_K,
                                l2_loop_order,
                                l1_loop_order,
                                l0_M_tiling_factor,
                                l0_N_tiling_factor,
                                l0_K_tiling_factor,
                            )
                            cycle_count = self.simulate(
                                self.computational_graph,
                                mapping,
                                pcb_module,
                            )
                            # end = time.time()
                            # if i % 1000 == 0:
                            #     print(f"{i} simulation time: {end-start}")
                            if cycle_count < min_cycle_count:
                                min_cycle_count = cycle_count
                                best_mapping = mapping
        elif compile_mode == "heuristic-GPU":
            i = 0
            for l2_tile_M in [64, 128, 256, 512, 1024, 2048]:
                for l2_tile_N in [l2_tile_M // 2, l2_tile_M, l2_tile_M * 2]:
                    if K <= 12288:
                        l2_K_tiling_factor_list = [1, 2, 4, 8]
                    else:
                        l2_K_tiling_factor_list = [
                            K // 1024,
                            K // 2048,
                            K // 4096,
                            K // 8192,
                        ]
                    for l2_K_tiling_factor in l2_K_tiling_factor_list:
                        l2_tile_K = ceil(
                            self.computational_graph.K / l2_K_tiling_factor
                        )
                        l2_tile_K = 2 ** floor(log2(l2_tile_K))
                        working_set_size = (
                            l2_tile_N * l2_tile_K
                            + l2_tile_M * l2_tile_K
                            + l2_tile_M * l2_tile_N
                        )
                        if (
                            working_set_size
                            > pcb_module.compute_module.l2_size
                            // self.data_type.word_size
                        ):
                            continue
                        elif (
                            working_set_size
                            <= pcb_module.compute_module.l2_size
                            // self.data_type.word_size
                            // 2
                        ):
                            is_l2_double_buffering = True
                        else:
                            is_l2_double_buffering = False

                        for l1_tile_M in [32, 64, 128, 256]:
                            if l1_tile_M > min(l2_tile_M, l2_tile_N):
                                continue
                            l1_tile_N = l1_tile_M
                            for l1_K_tiling_factor in [1, 2, 4, 8, 16, 32]:
                                l1_tile_K = ceil(l2_tile_K / l1_K_tiling_factor)
                                if (
                                    l1_tile_M * l1_tile_N
                                    + l1_tile_N * l1_tile_K
                                    + l1_tile_M * l1_tile_K
                                    > pcb_module.compute_module.core.SRAM_size
                                    // self.data_type.word_size
                                    // 2
                                ):
                                    continue
                                l2_loop_order = "knm"
                                l1_loop_order = "knm"
                                for (
                                    l0_M_tiling_factor,
                                    l0_N_tiling_factor,
                                    l0_K_tiling_factor,
                                ) in self.find_permutations(
                                    pcb_module.compute_module.core.systolic_array_count
                                ):
                                    i += 1
                                    start = time.time()
                                    mapping = self.Mapping(
                                        l2_tile_M,
                                        l2_tile_N,
                                        l2_tile_K,
                                        is_l2_double_buffering,
                                        l1_tile_M,
                                        l1_tile_N,
                                        l1_tile_K,
                                        l2_loop_order,
                                        l1_loop_order,
                                        l0_M_tiling_factor,
                                        l0_N_tiling_factor,
                                        l0_K_tiling_factor,
                                    )
                                    cycle_count = self.simulate(
                                        self.computational_graph,
                                        mapping,
                                        pcb_module,
                                    )
                                    end = time.time()
                                    # if i % 1000 == 0:
                                    #     print(f"{i} simulation time: {end-start}")
                                    if cycle_count < min_cycle_count:
                                        min_cycle_count = cycle_count
                                        best_mapping = mapping
            # print("total dse times:", i)
        elif compile_mode == "heuristic-TPU":
            l2_tile_M = self.computational_graph.M
            l2_tile_N = self.computational_graph.N
            l2_tile_K = self.computational_graph.K

            is_l2_double_buffering = True
            for l1_tile_M in [l2_tile_M, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
                if l1_tile_M > l2_tile_M * 2:
                    continue
                for l1_tile_N in [
                    l1_tile_M // 2,
                    l1_tile_M,
                    l1_tile_M * 2,
                    l1_tile_M * 8,
                    l1_tile_M * 16,
                    l1_tile_M * 64,
                    l1_tile_M * 128,
                    l1_tile_M * 256,
                ]:
                    if l1_tile_N > l2_tile_N:
                        continue
                    if l1_tile_N <= 0:
                        continue
                    l1_tile_K_max = (
                        pcb_module.compute_module.core.SRAM_size
                        // self.data_type.word_size
                        // 2
                        - l1_tile_M * l1_tile_N
                    ) // (l1_tile_M + l1_tile_N)
                    if l1_tile_K_max < 1:
                        continue
                    l1_tile_K = min(l1_tile_K_max, l2_tile_K)
                    l1_tile_K = floor(log2(l1_tile_K))
                    l1_tile_K = 2**l1_tile_K

                    l2_loop_order = "knm"
                    l1_loop_order = "knm"
                    for (
                        l0_M_tiling_factor,
                        l0_N_tiling_factor,
                        l0_K_tiling_factor,
                    ) in [(1, 2, 1)]:
                        mapping = self.Mapping(
                            l2_tile_M,
                            l2_tile_N,
                            l2_tile_K,
                            is_l2_double_buffering,
                            l1_tile_M,
                            l1_tile_N,
                            l1_tile_K,
                            l2_loop_order,
                            l1_loop_order,
                            l0_M_tiling_factor,
                            l0_N_tiling_factor,
                            l0_K_tiling_factor,
                        )
                        # mapping.display()
                        # start=time.time()
                        cycle_count = self.simulate(
                            self.computational_graph,
                            mapping,
                            pcb_module,
                        )
                        # end=time.time()
                        # print(f'simulation time: {end-start}')
                        if cycle_count < min_cycle_count:
                            min_cycle_count = cycle_count
                            best_mapping = mapping
        elif compile_mode == "heuristic-TPU-new":
            l2_tile_M = self.computational_graph.M
            l2_tile_N = self.computational_graph.N
            l2_tile_K = self.computational_graph.K

            is_l2_double_buffering = True
            for l1_tile_M in [l2_tile_M, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
                if l1_tile_M > l2_tile_M * 2:
                    continue
                for l1_tile_N in [
                    l1_tile_M // 2,
                    l1_tile_M,
                    l1_tile_M * 2,
                    l1_tile_M * 8,
                    l1_tile_M * 16,
                    l1_tile_M * 64,
                    l1_tile_M * 128,
                    l1_tile_M * 256,
                ]:
                    if l1_tile_N > l2_tile_N:
                        continue
                    if l1_tile_N <= 0:
                        continue
                    l1_tile_K_max = (
                        pcb_module.compute_module.core.SRAM_size
                        // self.data_type.word_size
                        // 2
                        - l1_tile_M * l1_tile_N
                    ) // (l1_tile_M + l1_tile_N)
                    if l1_tile_K_max < 1:
                        continue
                    l1_tile_K = min(l1_tile_K_max, l2_tile_K)
                    l1_tile_K = floor(log2(l1_tile_K))
                    l1_tile_K = 2**l1_tile_K

                    l2_loop_order = "knm"
                    l1_loop_order = "knm"
                    for (
                        l0_M_tiling_factor,
                        l0_N_tiling_factor,
                        l0_K_tiling_factor,
                    ) in [(1, 1, 1)]:
                        mapping = self.Mapping(
                            l2_tile_M,
                            l2_tile_N,
                            l2_tile_K,
                            is_l2_double_buffering,
                            l1_tile_M,
                            l1_tile_N,
                            l1_tile_K,
                            l2_loop_order,
                            l1_loop_order,
                            l0_M_tiling_factor,
                            l0_N_tiling_factor,
                            l0_K_tiling_factor,
                        )
                        # mapping.display()
                        # start=time.time()
                        cycle_count = self.simulate(
                            self.computational_graph,
                            mapping,
                            pcb_module,
                        )
                        # end=time.time()
                        # print(f'simulation time: {end-start}')
                        if cycle_count < min_cycle_count:
                            min_cycle_count = cycle_count
                            best_mapping = mapping
        elif compile_mode == "heuristic-SIMDRAM":
            self.latency = self.simdram_heuristic_tiling_v2(pcb_module)
        else:
            raise ValueError(f"compile_mode {compile_mode} not supported")
        self.best_mapping = best_mapping
        # if self.best_mapping is not None:
        #     self.best_mapping.display()
        self.best_cycle_count = min_cycle_count
        self.best_latency = min_cycle_count / pcb_module.compute_module.clock_freq
        self.latency = self.best_latency
        # self.best_mapping.display()
        return self.latency

    def compile_and_simulate_pimsab_gemv(
        self,
        pcb_module: Device,
        core_cnt_per_block,
        compile_mode: str = "heuristic-PIMSAB-sim-v2",
    ):
        if compile_mode == "heuristic-PIMSAB-sim-v2":
            #heuristic tiling
            M = self.computational_graph.M
            N = self.computational_graph.N
            K = self.computational_graph.K
            print(f"  M:{self.M}, K:{self.K}, N:{self.N}")
            assert M == 1
            num_tile = pcb_module.compute_module.tile_count
            num_block = pcb_module.compute_module.tile.arr_count
            num_row = pcb_module.compute_module.tile.arr_rows
            num_col = pcb_module.compute_module.tile.arr_cols
            dse_double_buffering = True

            K_tile_base = num_block #nblocks
            N_tile_base = num_col
            

            precision_input = self.data_type.word_size*8
            precision_accumulate = precision_input + 16
            tile_capacity = (num_row-precision_accumulate-2*precision_input)*num_col*num_block/8
            if dse_double_buffering:
                tile_capacity = tile_capacity // 2
            total_tile_base_size = (K_tile_base + N_tile_base + K_tile_base*N_tile_base)*self.data_type.word_size
            K_multiple = min(ceil(self.K/float(K_tile_base)), floor(tile_capacity/total_tile_base_size))#multiple K_tile_base
            K_tile = K_tile_base * K_multiple
            MK_tile_size = K_tile*self.data_type.word_size
            KN_and_N_tile_size = (K_tile*N_tile_base + N_tile_base)*self.data_type.word_size
            N_multiple = min(ceil(self.N/float(N_tile_base)) , floor((tile_capacity-MK_tile_size)/KN_and_N_tile_size))#rest of capacity for multiple N tile
            N_tile = N_multiple * N_tile_base

            M_tile = M
            

            
            print(f"tile size: {K_tile}, {N_tile}")
            pimsab_loop_order = "nkm"
            previous_m = 0
            previous_n = 0
            previous_k = 0
            total_latency = 0
            K_N_io_latency = max(K_tile * N_tile * self.data_type.word_size / (pcb_module.io_module.bandwidth / pcb_module.compute_module.tile_count), #1 tile can only use 1/tile_count of total dram bw 
                              K_tile * N_tile * self.data_type.word_size / (pcb_module.compute_module.noc.bandwidth))
            # M_K_io_latency = 0 due to already storing in sram.
            M_K_io_latency = 0
            M_N_io_latency = 0
            # print(f"tile dram bw: {pcb_module.io_module.bandwidth / pcb_module.compute_module.tile_count}")
            input_acc = self.data_type.word_size*8
            # print(f"accuracy: {input_acc}bit")
            accumulate_acc = input_acc+16
            debug = False
            tile_compute_latency ,_ = run_gemm(pcb_module,M_tile,K_tile,N_tile,input_acc,accumulate_acc, compute_only=True, debug=debug)
            print(f"K_N_io_latency: {K_N_io_latency}, M_K_io_latency: {M_K_io_latency}, M_N_io_latency: {M_N_io_latency}, tile_compute_latency:{tile_compute_latency}")
            
            M_t = M // M_tile
            N_t = N // N_tile
            K_t = K // K_tile
            M_remain = self.M % M_tile
            N_remain = self.N % N_tile
            K_remain = self.K % K_tile
            tile_latency = np.zeros(
                [ceil(M / M_tile), ceil(N / N_tile), ceil(K / K_tile)]
            )
            tile_shape_M = np.zeros([ceil(M / M_tile), ceil(N / N_tile), ceil(K / K_tile)])
            tile_shape_N = np.zeros([ceil(M / M_tile), ceil(N / N_tile), ceil(K / K_tile)])
            tile_shape_K = np.zeros([ceil(M / M_tile), ceil(N / N_tile), ceil(K / K_tile)])
            if M_t * N_t * K_t != 0:
                tile_latency[:M_t, :N_t, :K_t] = tile_compute_latency
                tile_shape_M[:M_t, :N_t, :K_t] = M_tile
                tile_shape_N[:M_t, :N_t, :K_t] = N_tile
                tile_shape_K[:M_t, :N_t, :K_t] = K_tile
            if M_remain != 0:
                tile_RKN_compute_latency ,_ = run_gemm(pcb_module,M_remain,K_tile,N_tile,input_acc,accumulate_acc, compute_only=True, debug=debug)
                tile_latency[-1, :N_t, :K_t] = tile_RKN_compute_latency
                tile_shape_M[-1, :N_t, :K_t] = M_remain
                tile_shape_N[-1, :N_t, :K_t] = N_tile
                tile_shape_K[-1, :N_t, :K_t] = K_tile
            if N_remain != 0:
                tile_MKR_compute_latency ,_ = run_gemm(pcb_module,M_tile,K_tile,N_remain,input_acc,accumulate_acc, compute_only=True, debug=debug)
                tile_latency[:M_t, -1, :K_t] = tile_MKR_compute_latency
                tile_shape_M[:M_t, -1, :K_t] = M_tile
                tile_shape_N[:M_t, -1, :K_t] = N_remain
                tile_shape_K[:M_t, -1, :K_t] = K_tile
            if K_remain != 0:
                tile_MRN_compute_latency ,_ = run_gemm(pcb_module,M_tile,K_remain,N_tile,input_acc,accumulate_acc, compute_only=True, debug=debug)
                tile_latency[:M_t, :N_t, -1] = tile_MRN_compute_latency
                tile_shape_M[:M_t, :N_t, -1] = M_tile
                tile_shape_N[:M_t, :N_t, -1] = N_tile
                tile_shape_K[:M_t, :N_t, -1] = K_remain
            if M_remain * N_remain != 0:
                tile_RKR_compute_latency ,_ = run_gemm(pcb_module,M_remain,K_tile,N_remain,input_acc,accumulate_acc, compute_only=True, debug=debug)
                tile_latency[-1, -1, :K_t] = tile_RKR_compute_latency
                tile_shape_M[-1, -1, :K_t] = M_remain
                tile_shape_N[-1, -1, :K_t] = N_remain
                tile_shape_K[-1, -1, :K_t] = K_tile
            if M_remain * K_remain != 0:
                tile_RRN_compute_latency ,_ = run_gemm(pcb_module,M_remain,K_remain,N_tile,input_acc,accumulate_acc, compute_only=True, debug=debug)
                tile_latency[-1, :N_t, -1] = tile_RRN_compute_latency
                tile_shape_M[-1, :N_t, -1] = M_remain
                tile_shape_N[-1, :N_t, -1] = N_tile
                tile_shape_K[-1, :N_t, -1] = K_remain
            if N_remain * K_remain != 0:
                tile_MRR_compute_latency ,_ = run_gemm(pcb_module,M_tile,K_remain,N_remain,input_acc,accumulate_acc, compute_only=True, debug=debug)
                tile_latency[:M_t, -1, -1] = tile_MRR_compute_latency
                tile_shape_M[:M_t, -1, -1] = M_tile
                tile_shape_N[:M_t, -1, -1] = N_remain
                tile_shape_K[:M_t, -1, -1] = K_remain
            if M_remain * N_remain * K_remain != 0:
                tile_RRR_compute_latency ,_ = run_gemm(pcb_module,M_remain,K_remain,N_remain,input_acc,accumulate_acc, compute_only=True, debug=debug)
                tile_latency[-1, -1, -1] = tile_RRR_compute_latency
                tile_shape_M[-1, -1, -1] = M_remain
                tile_shape_N[-1, -1, -1] = N_remain
                tile_shape_K[-1, -1, -1] = K_remain

            # with np.printoptions(threshold=np.inf):
            #     print(tile_latency)
            for m, n, k in self.generate_tile_loops(
                ceil(M / M_tile),
                ceil(N / N_tile),
                ceil(K / K_tile),    
                pimsab_loop_order,
            ):
                if m == 0 and n == 0 and k == 0:
                    #load data for the first tile. input and output stays in sram
                    total_latency += K_N_io_latency
                    continue

                
                #capacity per pimsab tile
                tile_capacity = (num_row-precision_accumulate-precision_input)*num_col*num_block/8
                M_tile = tile_shape_M[m,n,k]
                N_tile = tile_shape_N[m,n,k]
                K_tile = tile_shape_K[m,n,k]
                # print(f"m{m},k{k},n{n},M_tile: {M_tile}, K_tile: {K_tile}, N_tile: {N_tile}")
                M_K_tile_size = M_tile*K_tile*self.data_type.word_size #assume blockwise-unbroadcasted (tight) data layout
                K_N_tile_size = K_tile*N_tile*self.data_type.word_size
                M_N_tile_size = M_tile*N_tile*self.data_type.word_size
                assert M_K_tile_size + K_N_tile_size + M_N_tile_size <= tile_capacity
                K_N_io_latency = max(K_tile * N_tile * self.data_type.word_size / (pcb_module.io_module.bandwidth / pcb_module.compute_module.tile_count), #1 tile can only use 1/tile_count of total dram bw 
                              K_tile * N_tile * self.data_type.word_size / (pcb_module.compute_module.noc.bandwidth))
                # print(f"m{m},k{k},n{n}, K_N_io_latency: {K_N_io_latency}")
                # M_K_io_latency = 0 due to always storing in sram.
                M_K_io_latency = 0
                M_N_io_latency = 0

                # determine possible double buffering
                double_buffering = False
                # current tile read latency
                if m == previous_m and k == previous_k:
                    current_tile_read_latency = K_N_io_latency
                    if M_K_tile_size + 2*K_N_tile_size +2*M_N_tile_size<=tile_capacity:
                        double_buffering = True
                elif n == previous_n and k == previous_k:
                    current_tile_read_latency = M_K_io_latency
                    if 2*M_K_tile_size + K_N_tile_size + 2*M_N_tile_size<=tile_capacity:
                        double_buffering = True
                else:
                    current_tile_read_latency = (
                        M_K_io_latency + K_N_io_latency
                    )
                    if m==previous_m and n==previous_n and 2*M_K_tile_size + 2*K_N_tile_size + M_N_tile_size<=tile_capacity:
                        double_buffering = True
                    elif 2*M_K_tile_size + 2*K_N_tile_size + 2*M_N_tile_size<=tile_capacity:
                        double_buffering = True
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

                # read current tile, compute previous tile, write previous tile
                if double_buffering:  # pipelined
                    
                    tile_total_latency = (
                        max(
                            current_tile_read_latency, previous_tile_compute_latency
                        )
                        + previous_tile_write_latency
                    )
                    total_latency += tile_total_latency
                    # print(f"m{m},k{k},n{n},tile_total_latency: {tile_total_latency}")
                else:  # non-pipelined
                    total_latency += (
                        current_tile_read_latency
                        + previous_tile_compute_latency
                        + previous_tile_write_latency
                    )

                previous_m = m
                previous_n = n
                previous_k = k
                # print(f"m{m},k{k},n{n},total_latency: {total_latency}")
            # compute and write last tile
            total_latency += (
                M_N_io_latency
                + tile_compute_latency
            )
            

            # if previous_k > 0:
            #     total_cycle_count += ceil(l2_tiles[-1, -1, -1].K_reduction_cycle_count)
            print(f"gemv latency: {total_latency}")
            return total_latency
        
    def compile_and_simulate_pimsab(
        self,
        pcb_module: Device,
        compile_mode: str = "exhaustive",   
    ):
        assert pcb_module.type == "pimsab"
        M = self.computational_graph.M
        N = self.computational_graph.N
        K = self.computational_graph.K
        if self.M ==1 or self.N == 1:
            return self.compile_and_simulate_pimsab_gemv(pcb_module, 1, compile_mode)
        if compile_mode == "heuristic-PIMSAB":
            print(f"  matmul: {self.input1_shape}, {self.input2_shape}, {self.output_shape}")
            print(f"  M:{self.M}, K:{self.K}, N:{self.N}")
            self.latency = self.roofline_model(pcb_module)
            return self.latency
        
        elif compile_mode == "heuristic-PIMSAB-sim":
            print(f"  matmul: {self.input1_shape}, {self.input2_shape}, {self.output_shape}")
            print(f"  M:{self.M}, K:{self.K}, N:{self.N}")
            M_tile_base = 120
            K_tile_base = 256 #nblocks
            N_tile_base = 256
            N_tile = N_tile_base
            tile_capacity = (256-32-16)*256*256/8
            K_multiple = min(ceil(self.K/float(K_tile_base)), floor((tile_capacity-256*256*self.data_type.word_size)/(K_tile_base*N_tile_base*self.data_type.word_size)))
            K_tile = K_tile_base * K_multiple
            M_multiple = min(ceil(self.M/float(M_tile_base)) , floor((tile_capacity-K_tile*N_tile*self.data_type.word_size)/256/256/self.data_type.word_size))
            M_tile = M_multiple * M_tile_base
            print(f"tile size: {M_tile}, {K_tile}, {N_tile}")
            latency,_ = run_gemm(M_tile,K_tile,N_tile,debug=True)
            iterations = ceil(self.M/M_tile)*ceil(self.N/N_tile)*ceil(self.K/K_tile)
            self.latency = latency * iterations
            print(f"iterations:{iterations}")
            print(f"total latency:{self.latency}")
            return self.latency

        elif compile_mode == "heuristic-PIMSAB-sim-v2":
            
            print(f"  matmul: {self.input1_shape}, {self.input2_shape}, {self.output_shape}")
            print(f"  M:{self.M}, K:{self.K}, N:{self.N}")


            num_tile = pcb_module.compute_module.tile_count
            num_block = pcb_module.compute_module.tile.arr_count
            num_row = pcb_module.compute_module.tile.arr_rows
            num_col = pcb_module.compute_module.tile.arr_cols
            dse_double_buffering = True

            M_tile_base = num_tile
            K_tile_base = num_block #nblocks
            N_tile_base = num_col
            
             #crams
            N_tile = N_tile_base
            precision_input = self.data_type.word_size
            precision_accumulate = precision_input + 16
            tile_capacity = (num_row-precision_accumulate-precision_input)*num_col*num_block/8
            K_multiple = min(ceil(self.K/float(K_tile_base)), floor((tile_capacity-num_block*num_col*self.data_type.word_size)/(K_tile_base*N_tile_base*self.data_type.word_size)))#leaving 8 rows, then use all for KN
            K_tile = K_tile_base * K_multiple
            M_multiple = min(ceil(self.M/float(M_tile_base)) , floor((tile_capacity-K_tile*N_tile*self.data_type.word_size)/(K_tile+N_tile)/self.data_type.word_size))#rest of capacity for MK and MN
            M_tile = M_multiple * M_tile_base

            if dse_double_buffering:
                M_tile = M_tile //2
                K_tile = K_tile //2
            
            print(f"tile size: {M_tile}, {K_tile}, {N_tile}")
            pimsab_loop_order = "nkm"
            previous_m = 0
            previous_n = 0
            previous_k = 0
            total_latency = 0
            K_N_io_latency = (K_tile * N_tile * self.data_type.word_size / pcb_module.io_module.bandwidth
                             + K_tile * N_tile * self.data_type.word_size / (1024*pcb_module.compute_module.clock_freq/8))
            M_K_io_latency = M_tile * K_tile * self.data_type.word_size / pcb_module.io_module.bandwidth
            M_N_io_latency = M_tile * N_tile * self.data_type.word_size / pcb_module.io_module.bandwidth

            input_acc = self.data_type.word_size*8
            print(f"accuracy: {input_acc}bit")
            accumulate_acc = input_acc+16
            debug = False
            tile_compute_latency ,_ = run_gemm(pcb_module,M_tile,K_tile,N_tile,input_acc,accumulate_acc, compute_only=True, debug=debug)
            print(f"K_N_io_latency: {K_N_io_latency}, M_K_io_latency: {M_K_io_latency}, M_N_io_latency: {M_N_io_latency}, tile_compute_latency:{tile_compute_latency}")
            
            M_t = M // M_tile
            N_t = N // N_tile
            K_t = K // K_tile
            M_remain = self.M % M_tile
            N_remain = self.N % N_tile
            K_remain = self.K % K_tile
            tile_latency = np.zeros(
                [ceil(M / M_tile), ceil(N / N_tile), ceil(K / K_tile)]
            )
            if M_t * N_t * K_t != 0:
                tile_latency[:M_t, :N_t, :K_t] = tile_compute_latency
            if M_remain != 0:
                tile_RKN_compute_latency ,_ = run_gemm(pcb_module,M_remain,K_tile,N_tile,input_acc,accumulate_acc, compute_only=True, debug=debug)
                tile_latency[-1, :N_t, :K_t] = tile_RKN_compute_latency
            if N_remain != 0:
                tile_MKR_compute_latency ,_ = run_gemm(pcb_module,M_tile,K_tile,N_remain,input_acc,accumulate_acc, compute_only=True, debug=debug)
                tile_latency[:M_t, -1, :K_t] = tile_MKR_compute_latency
            if K_remain != 0:
                tile_MRN_compute_latency ,_ = run_gemm(pcb_module,M_tile,K_remain,N_tile,input_acc,accumulate_acc, compute_only=True, debug=debug)
                tile_latency[:M_t, :N_t, -1] = tile_MRN_compute_latency
            if M_remain * N_remain != 0:
                tile_RKR_compute_latency ,_ = run_gemm(pcb_module,M_remain,K_tile,N_remain,input_acc,accumulate_acc, compute_only=True, debug=debug)
                tile_latency[-1, -1, :K_t] = tile_RKR_compute_latency
            if M_remain * K_remain != 0:
                tile_RRN_compute_latency ,_ = run_gemm(pcb_module,M_remain,K_remain,N_tile,input_acc,accumulate_acc, compute_only=True, debug=debug)
                tile_latency[-1, :N_t, -1] = tile_RRN_compute_latency
            if N_remain * K_remain != 0:
                tile_MRR_compute_latency ,_ = run_gemm(pcb_module,M_tile,K_remain,N_remain,input_acc,accumulate_acc, compute_only=True, debug=debug)
                tile_latency[:M_t, -1, -1] = tile_MRR_compute_latency
            if M_remain * N_remain * K_remain != 0:
                tile_RRR_compute_latency ,_ = run_gemm(pcb_module,M_remain,K_remain,N_remain,input_acc,accumulate_acc, compute_only=True, debug=debug)
                tile_latency[-1, -1, -1] = tile_RRR_compute_latency

            # with np.printoptions(threshold=np.inf):
            #     print(tile_latency)
            for m, n, k in self.generate_tile_loops(
                ceil(M / M_tile),
                ceil(N / N_tile),
                ceil(K / K_tile),    
                pimsab_loop_order,
            ):
                if m == 0 and n == 0 and k == 0:
                    #load data for first tile
                    total_latency += M_N_io_latency + M_K_io_latency + K_N_io_latency
                    continue

                
                #capacity per pimsab tile
                tile_capacity = (num_row-32-16)*num_col*num_block/8
                M_K_tile_size = M_tile/num_tile*K_tile*self.data_type.word_size #assume blockwise-unbroadcasted (tight) data layout
                K_N_tile_size = K_tile*N_tile*self.data_type.word_size
                M_N_tile_size = M_tile/num_tile*N_tile*self.data_type.word_size
                assert M_K_tile_size + K_N_tile_size + M_N_tile_size <= tile_capacity

                # determine possible double buffering
                double_buffering = False
                # current tile read latency
                if m == previous_m and k == previous_k:
                    current_tile_read_latency = K_N_io_latency
                    if M_K_tile_size + 2*K_N_tile_size +2*M_N_tile_size<=tile_capacity:
                        double_buffering = True
                elif n == previous_n and k == previous_k:
                    current_tile_read_latency = M_K_io_latency
                    if 2*M_K_tile_size + K_N_tile_size + 2*M_N_tile_size<=tile_capacity:
                        double_buffering = True
                else:
                    current_tile_read_latency = (
                        M_K_io_latency + K_N_io_latency
                    )
                    if m==previous_m and n==previous_n and 2*M_K_tile_size + 2*K_N_tile_size + M_N_tile_size<=tile_capacity:
                        double_buffering = True
                    elif 2*M_K_tile_size + 2*K_N_tile_size + 2*M_N_tile_size<=tile_capacity:
                        double_buffering = True
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
        else:
            raise ValueError(f"compile_mode {compile_mode} not supported")
    
    def simdram_heuristic_gemm_Siyuan(self, pcb_module: Device):
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


        arr_tile_N, arr_tile_M = self.find_tile_N_M(num_col_per_array)
        factor_N = ceil(N / arr_tile_N)
        factor_M = ceil(M / arr_tile_M)
        # find nearest power of 2 of tile_K that satisfy the row limit
        arr_tile_K, accum_bits = self.find_tile_K(num_row)
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

        K_N_io_latency = K_tile * N_tile * self.data_type.word_size * num_array / pcb_module.io_module.bandwidth #move K_tile x N_tile to compute-dram, consider duplication across arrays
        M_K_io_latency = M_tile * K_tile * self.data_type.word_size * num_bank / pcb_module.io_module.bandwidth #move K_tile x K_tile to compute-dram, consider duplication across banks
        M_N_io_latency = M_tile * N_tile * self.data_type.word_size / pcb_module.io_module.bandwidth #move M_tile x N_tile to compute-dram, no duplication as K is reduction axis
    
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
        tile_compute_latency += M_tile * N_tile * num_device * self.data_type.word_size / pcb_module.io_module.bandwidth

        #per array latency for K_remain
        add_latency_per_array_remain = arr_K_remain * simdram_op_latency_dict[self.data_type.name]['add']
        mul_latency_per_array_remain = arr_K_remain * simdram_op_latency_dict[self.data_type.name]['mul']
        # Add extra accumulation latency
        tile_compute_latency_remain = (add_latency_per_array_remain + mul_latency_per_array_remain + simdram_op_latency_dict['fp32']['add'] + simdram_op_latency_dict['fp32']['mul'])*1e-9
        # reduction across devices
        tile_compute_latency += M_remain * N_remain * num_device * self.data_type.word_size / pcb_module.io_module.bandwidth
        
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
        # M_tile x K_tile is moved from somewhere else to compute-enabled DRAM and duplicated across banks to enable N tiling
        # K_tile x N_tile is already in compute-enabled DRAM and already duplicated across arrays to enable M tiling
        # M_tile x N_tile is moved from somewhere else to compute-enabled DRAM, does not need any duplication across K dimension (mapped to devices) as K is reduction axis
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
            return self.simdram_heuristic_gemm_Siyuan(pcb_module)
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
        if pcb_module.type=="systolic":
            return self.compile_and_simulate_systolic(pcb_module, compile_mode)
        elif pcb_module.type=="pimsab":
            return self.compile_and_simulate_pimsab(pcb_module, compile_mode)
        elif pcb_module.type == "simdram":
            return self.compile_and_simulate_simdram(pcb_module, compile_mode)
        else:
            raise ValueError("Unsupported device type!")
        
    def simulate(
        self,
        computational_graph: ComputationalGraph,
        mapping: Mapping,
        pcb_module: Device,
    ) -> int:
        if self.look_up_table is None:
            self.look_up_table = pd.read_csv(
                f"./systolic_array_model/look_up_table_{pcb_module.compute_module.core.systolic_array.array_height}_{pcb_module.compute_module.core.systolic_array.array_width}.csv",
                header=None,
                names=[
                    "M",
                    "N",
                    "K",
                    "ArrayHeight",
                    "ArrayWidth",
                    "Dataflow",
                    "cycle_count",
                    "util_rate",
                ],
            )
            self.look_up_table.drop_duplicates(
                inplace=True,
                subset=["M", "N", "K", "ArrayHeight", "ArrayWidth", "Dataflow"],
            )
            # self.look_up_table.reset_index(drop=True, inplace=True)
            # self.look_up_table.to_csv(
            #     f"./systolic_array_model/look_up_table_{pcb_module.compute_module.core.systolic_array.array_height}_{pcb_module.compute_module.core.systolic_array.array_width}.csv",
            #     header=False,
            #     index=False,
            # )
            self.look_up_table.set_index(
                ["M", "N", "K", "ArrayHeight", "ArrayWidth", "Dataflow"],
                inplace=True,
            )
        # print(self.look_up_table)
        # print(self.look_up_table.loc[(32, 16, 256, 16, 16, 'os'), "cycle_count"
        #                              ].item())
        # print('sdfsdfsdfsd')
        # exit()
        M = computational_graph.M
        N = computational_graph.N
        K = computational_graph.K
        data_type = computational_graph.data_type

        l2_tile_M = mapping.l2_tile_M
        l2_tile_N = mapping.l2_tile_N
        l2_tile_K = mapping.l2_tile_K

        if mapping.is_l2_double_buffering:
            assert (
                l2_tile_M * l2_tile_N + l2_tile_N * l2_tile_K + l2_tile_M * l2_tile_K
                <= pcb_module.compute_module.l2_size // self.data_type.word_size // 2
            )
        else:
            assert (
                l2_tile_M * l2_tile_N + l2_tile_N * l2_tile_K + l2_tile_M * l2_tile_K
                <= pcb_module.compute_module.l2_size // self.data_type.word_size
            )

        M_l2_t = M // l2_tile_M
        N_l2_t = N // l2_tile_N
        K_l2_t = K // l2_tile_K
        M_remain = M % l2_tile_M
        N_remain = N % l2_tile_N
        K_remain = K % l2_tile_K

        l2_tiles = np.empty(
            [ceil(M / l2_tile_M), ceil(N / l2_tile_N), ceil(K / l2_tile_K)],
            dtype=self.L2TileSimulator,
        )
        # print('-'*20)
        # print(l2_tiles.shape)
        if M_l2_t * N_l2_t * K_l2_t != 0:
            l2_tiles[:M_l2_t, :N_l2_t, :K_l2_t] = self.L2TileSimulator(
                l2_tile_M,
                l2_tile_N,
                l2_tile_K,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
        if M_remain != 0:
            l2_tiles[-1, :N_l2_t, :K_l2_t] = self.L2TileSimulator(
                M_remain,
                l2_tile_N,
                l2_tile_K,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
        if N_remain != 0:
            l2_tiles[:M_l2_t, -1, :K_l2_t] = self.L2TileSimulator(
                l2_tile_M,
                N_remain,
                l2_tile_K,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
        if K_remain != 0:
            l2_tiles[:M_l2_t, :N_l2_t, -1] = self.L2TileSimulator(
                l2_tile_M,
                l2_tile_N,
                K_remain,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
        if M_remain * N_remain != 0:
            l2_tiles[-1, -1, :K_l2_t] = self.L2TileSimulator(
                M_remain,
                N_remain,
                l2_tile_K,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
        if M_remain * K_remain != 0:
            l2_tiles[-1, :N_l2_t, -1] = self.L2TileSimulator(
                M_remain,
                l2_tile_N,
                K_remain,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
        if N_remain * K_remain != 0:
            l2_tiles[:M_l2_t, -1, -1] = self.L2TileSimulator(
                l2_tile_M,
                N_remain,
                K_remain,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
        if M_remain * N_remain * K_remain != 0:
            l2_tiles[-1, -1, -1] = self.L2TileSimulator(
                M_remain,
                N_remain,
                K_remain,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )

        total_cycle_count = 0
        total_cycle_count += (
            l2_tiles[0, 0, 0].M_K_io_cycle_count + l2_tiles[0, 0, 0].K_N_io_cycle_count
        )

        previous_m = 0
        previous_n = 0
        previous_k = 0

        for m, n, k in self.generate_tile_loops(
            ceil(M / l2_tile_M),
            ceil(N / l2_tile_N),
            ceil(K / l2_tile_K),
            mapping.l2_loop_order,
        ):
            if m == 0 and n == 0 and k == 0:
                continue

            l2_tile = l2_tiles[m, n, k]
            previous_l2_tile = l2_tiles[previous_m, previous_n, previous_k]

            # current tile read latency
            if m == previous_m and k == previous_k:
                current_tile_read_cycle_count = l2_tile.K_N_io_cycle_count
            elif n == previous_n and k == previous_k:
                current_tile_read_cycle_count = l2_tile.M_K_io_cycle_count
            else:
                current_tile_read_cycle_count = (
                    l2_tile.M_K_io_cycle_count + l2_tile.K_N_io_cycle_count
                )
            if k > 0 and not (m == previous_m and n == previous_n):
                current_tile_read_cycle_count += l2_tile.M_N_io_cycle_count
            # previous tile compute latency
            previous_tile_compute_cycle_count = previous_l2_tile.compute_cycle_count
            if k > 0:
                previous_tile_compute_cycle_count += (
                    previous_l2_tile.K_reduction_cycle_count
                )
            # previous tile write latency
            if m == previous_m and n == previous_n:
                previous_tile_write_cycle_count = 0
            else:
                previous_tile_write_cycle_count = previous_l2_tile.M_N_io_cycle_count

            # read current tile, compute previous tile, write previous tile
            if mapping.is_l2_double_buffering:  # pipelined
                total_cycle_count += (
                    max(
                        current_tile_read_cycle_count, previous_tile_compute_cycle_count
                    )
                    + previous_tile_write_cycle_count
                )
            else:  # non-pipelined
                total_cycle_count += (
                    current_tile_read_cycle_count
                    + previous_tile_compute_cycle_count
                    + previous_tile_write_cycle_count
                )

            previous_m = m
            previous_n = n
            previous_k = k

        # compute and write last tile
        total_cycle_count += (
            l2_tiles[-1, -1, -1].M_N_io_cycle_count
            + l2_tiles[-1, -1, -1].compute_cycle_count
        )

        if previous_k > 0:
            total_cycle_count += ceil(l2_tiles[-1, -1, -1].K_reduction_cycle_count)

        return total_cycle_count #+ ceil(
        # pcb_module.io_module.latency * 2 * pcb_module.compute_module.clock_freq
        # )

    class L2TileSimulator:
        def __init__(
            self,
            M: int,
            N: int,
            K: int,
            data_type: DataType,
            mapping: "Matmul.Mapping",
            pcb_module: Device,
            look_up_table: pd.DataFrame,
        ):
            # print(f'L2 tile: {M} {N} {K}')
            self.M = M
            self.N = N
            self.K = K
            self.K_reduction_cycle_count = ceil(
                M * N / pcb_module.compute_module.total_vector_flops_per_cycle
            ) + 2 * ceil(
                M
                * N
                * data_type.word_size
                / pcb_module.compute_module.l2_bandwidth_per_cycle
            )
            self.K_reduction_io_count = 2 * M * N * data_type.word_size
            self.M_K_io_cycle_count = self.simulate_l2_tile_io_cycle_count(
                M, K, data_type, pcb_module
            )
            self.K_N_io_cycle_count = self.simulate_l2_tile_io_cycle_count(
                K, N, data_type, pcb_module
            )
            self.M_N_io_cycle_count = self.simulate_l2_tile_io_cycle_count(
                M, N, data_type, pcb_module
            )
            self.compute_cycle_count = self.simulate_l2_tile_compute_cycle_count(
                M, N, K, data_type, mapping, pcb_module, look_up_table
            )

        def simulate_l2_tile_io_cycle_count(
            self, M: int, N: int, data_type: DataType, chiplet_module: Device
        ):
            return ceil(
                M
                * N
                * data_type.word_size
                / (
                    chiplet_module.io_module.bandwidth
                    / chiplet_module.compute_module.clock_freq
                )
            )

        def simulate_l2_tile_compute_cycle_count(
            self,
            M: int,
            N: int,
            K: int,
            data_type: DataType,
            mapping: "Matmul.Mapping",
            chiplet_module: Device,
            look_up_table: pd.DataFrame,
        ) -> int:
            l1_tile_M = mapping.l1_tile_M
            l1_tile_N = mapping.l1_tile_N
            l1_tile_K = mapping.l1_tile_K

            M_l1_t = M // l1_tile_M
            N_l1_t = N // l1_tile_N
            K_l1_t = K // l1_tile_K
            M_remain = M % l1_tile_M
            N_remain = N % l1_tile_N
            K_remain = K % l1_tile_K

            l1_tiles = np.empty(
                [ceil(M / l1_tile_M), ceil(N / l1_tile_N), ceil(K / l1_tile_K)],
                dtype=Matmul.L1TileSimulator,
            )
            # fill in the full tiles
            if M_l1_t * N_l1_t * K_l1_t != 0:
                l1_tiles[:M_l1_t, :N_l1_t, :K_l1_t] = Matmul.L1TileSimulator(
                    l1_tile_M,
                    l1_tile_N,
                    l1_tile_K,
                    data_type,
                    mapping,
                    chiplet_module,
                    look_up_table,
                )
            # M dimension is remaining
            if M_remain != 0:
                l1_tiles[-1, :N_l1_t, :K_l1_t] = Matmul.L1TileSimulator(
                    M_remain,
                    l1_tile_N,
                    l1_tile_K,
                    data_type,
                    mapping,
                    chiplet_module,
                    look_up_table,
                )
            # N dimension is remaining
            if N_remain != 0:
                l1_tiles[:M_l1_t, -1, :K_l1_t] = Matmul.L1TileSimulator(
                    l1_tile_M,
                    N_remain,
                    l1_tile_K,
                    data_type,
                    mapping,
                    chiplet_module,
                    look_up_table,
                )
            # K dimension is remaining
            if K_remain != 0:
                l1_tiles[:M_l1_t, :N_l1_t, -1] = Matmul.L1TileSimulator(
                    l1_tile_M,
                    l1_tile_N,
                    K_remain,
                    data_type,
                    mapping,
                    chiplet_module,
                    look_up_table,
                )
            if M_remain * N_remain != 0:
                l1_tiles[-1, -1, :K_l1_t] = Matmul.L1TileSimulator(
                    M_remain,
                    N_remain,
                    l1_tile_K,
                    data_type,
                    mapping,
                    chiplet_module,
                    look_up_table,
                )
            if M_remain * K_remain != 0:
                l1_tiles[-1, :N_l1_t, -1] = Matmul.L1TileSimulator(
                    M_remain,
                    l1_tile_N,
                    K_remain,
                    data_type,
                    mapping,
                    chiplet_module,
                    look_up_table,
                )
            if N_remain * K_remain != 0:
                l1_tiles[:M_l1_t, -1, -1] = Matmul.L1TileSimulator(
                    l1_tile_M,
                    N_remain,
                    K_remain,
                    data_type,
                    mapping,
                    chiplet_module,
                    look_up_table,
                )
            if M_remain * N_remain * K_remain != 0:
                l1_tiles[-1, -1, -1] = Matmul.L1TileSimulator(
                    M_remain,
                    N_remain,
                    K_remain,
                    data_type,
                    mapping,
                    chiplet_module,
                    look_up_table,
                )
            # Fill in the actual M x K matrix dimensions, and handling edge cases
            M_K_tile_size = np.zeros(
                [ceil(M / l1_tile_M), ceil(K / l1_tile_K)], dtype=int
            )
            M_K_tile_size[:M_l1_t, :K_l1_t] = l1_tile_M * l1_tile_K
            if M_remain > 0:
                M_K_tile_size[-1, :K_l1_t] = M_remain * l1_tile_K
            if K_remain > 0:
                M_K_tile_size[:M_l1_t, -1] = l1_tile_M * K_remain
            if M_remain > 0 and K_remain > 0:
                M_K_tile_size[-1, -1] = M_remain * K_remain
            # Fill in the actual K x N matrix dimensions, and handling edge cases
            K_N_tile_size = np.zeros(
                [ceil(K / l1_tile_K), ceil(N / l1_tile_N)], dtype=int
            )
            K_N_tile_size[:K_l1_t, :N_l1_t] = l1_tile_K * l1_tile_N
            if K_remain > 0:
                K_N_tile_size[-1, :N_l1_t] = K_remain * l1_tile_N
            if N_remain > 0:
                K_N_tile_size[:K_l1_t, -1] = l1_tile_K * N_remain
            if K_remain > 0 and N_remain > 0:
                K_N_tile_size[-1, -1] = K_remain * N_remain
            # Fill in the actual M x N matrix dimensions, and handling edge cases
            M_N_tile_size = np.zeros(
                [ceil(M / l1_tile_M), ceil(N / l1_tile_N)], dtype=int
            )
            M_N_tile_size[:M_l1_t, :N_l1_t] = l1_tile_M * l1_tile_N
            if M_remain > 0:
                M_N_tile_size[-1, :N_l1_t] = M_remain * l1_tile_N
            if N_remain > 0:
                M_N_tile_size[:M_l1_t, -1] = l1_tile_M * N_remain
            if M_remain > 0 and N_remain > 0:
                M_N_tile_size[-1, -1] = M_remain * N_remain

            total_cycle_count = 0
            previous_batch_Read_M_K = np.zeros(
                [ceil(M / l1_tile_M), ceil(K / l1_tile_K)], dtype=bool
            )
            previous_batch_Read_K_N = np.zeros(
                [ceil(K / l1_tile_K), ceil(N / l1_tile_N)], dtype=bool
            )
            previous_batch_Read_M_N = np.zeros(
                [ceil(M / l1_tile_M), ceil(N / l1_tile_N)], dtype=bool
            )
            previous_batch_Write_M_N = np.zeros(
                [ceil(M / l1_tile_M), ceil(N / l1_tile_N)], dtype=bool
            )
            previous_batch_compute_cycle_count = 0
            active_l1_tile_list = []
            for m, n, k in Matmul.generate_tile_loops(
                ceil(M / l1_tile_M),
                ceil(N / l1_tile_N),
                ceil(K / l1_tile_K),
                mapping.l1_loop_order,
            ):
                # Map current work to an L1 tile, set as active
                active_l1_tile_list.append((m, n, k, l1_tiles[m, n, k]))
                # if current M x K x N is the last round of workgroup, no further check is required
                if (
                    m == ceil(M / l1_tile_M) - 1
                    and n == ceil(N / l1_tile_N) - 1
                    and k == ceil(K / l1_tile_K) - 1
                ):
                    pass
                elif (
                    # continue add workloads until L1 tiles are fulfilled.
                    # before L1 tiles are saturated, no computation simulation is performed.
                    len(active_l1_tile_list) < chiplet_module.compute_module.core_count
                ):
                    continue

                assert (
                    len(active_l1_tile_list) <= chiplet_module.compute_module.core_count
                )
                current_batch_Read_M_K = np.zeros(
                    [ceil(M / l1_tile_M), ceil(K / l1_tile_K)], dtype=bool
                )
                current_batch_Read_K_N = np.zeros(
                    [ceil(K / l1_tile_K), ceil(N / l1_tile_N)], dtype=bool
                )
                current_batch_Read_M_N = np.zeros(
                    [ceil(M / l1_tile_M), ceil(N / l1_tile_N)], dtype=bool
                )
                current_batch_Write_M_N = np.zeros(
                    [ceil(M / l1_tile_M), ceil(N / l1_tile_N)], dtype=bool
                )

                current_batch_compute_cycle_count = 0
                # simulate computation for every L1 active tiles
                for i in range(len(active_l1_tile_list)):
                    temp_m, temp_n, temp_k, temp_l1_tile = active_l1_tile_list[i]
                    current_batch_Read_M_K[temp_m, temp_k] = 1
                    current_batch_Read_K_N[temp_k, temp_n] = 1
                    current_batch_Read_M_N[temp_m, temp_n] = temp_k > 0
                    current_batch_Write_M_N[temp_m, temp_n] = 1
                    temp_l1_tile_compute_cycle_count = temp_l1_tile.compute_cycle_count
                    if temp_k > 0:
                        temp_l1_tile_compute_cycle_count += ceil(
                            temp_l1_tile.M
                            * temp_l1_tile.N
                            / chiplet_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
                        )
                    current_batch_compute_cycle_count = max(
                        current_batch_compute_cycle_count,
                        temp_l1_tile_compute_cycle_count,
                    )

                # if one output tile in this batch shares input/output with another output tile in the previous batch, assign them to the same core to avoid data movement
                # note that of the three input matrix mk, kn, mn, at most one of them can be the same if we change m,n,k
                current_batch_M_K_read_count = np.sum(
                    (current_batch_Read_M_K * (~previous_batch_Read_M_K))
                    * M_K_tile_size
                )
                current_batch_K_N_read_count = np.sum(
                    (current_batch_Read_K_N * (~previous_batch_Read_K_N))
                    * K_N_tile_size
                )
                current_batch_M_N_read_count = np.sum(
                    (current_batch_Read_M_N
                        * (~(previous_batch_Read_M_N + previous_batch_Write_M_N))
                    )
                    * M_N_tile_size
                )
                previous_batch_M_N_write_count = np.sum(
                    (previous_batch_Write_M_N * (~current_batch_Read_M_N))
                    * M_N_tile_size
                )

                # read current batch while compute and write previous batch
                current_batch_read_count = (
                    current_batch_M_K_read_count
                    + current_batch_K_N_read_count
                    + current_batch_M_N_read_count
                )
                current_batch_read_cycle_count = ceil(
                    current_batch_read_count
                    * chiplet_module.compute_module.core.systolic_array.input_word_size
                    / chiplet_module.compute_module.l2_bandwidth_per_cycle
                )
                prvious_batch_write_cycle_count = ceil(
                    previous_batch_M_N_write_count
                    * chiplet_module.compute_module.core.systolic_array.output_word_size
                    / chiplet_module.compute_module.l2_bandwidth_per_cycle
                )

                total_cycle_count += (
                    max(
                        current_batch_read_cycle_count,
                        previous_batch_compute_cycle_count,
                    )
                    + prvious_batch_write_cycle_count
                )

                previous_batch_compute_cycle_count = current_batch_compute_cycle_count
                previous_batch_Read_M_K = copy.deepcopy(current_batch_Read_M_K)
                previous_batch_Read_K_N = copy.deepcopy(current_batch_Read_K_N)
                previous_batch_Read_M_N = copy.deepcopy(current_batch_Read_M_N)
                previous_batch_Write_M_N = copy.deepcopy(current_batch_Write_M_N)

                active_l1_tile_list = []

            # last batch's compute and write
            total_cycle_count += previous_batch_compute_cycle_count + ceil(
                np.sum(previous_batch_Write_M_N * M_N_tile_size)
                * data_type.word_size
                / chiplet_module.compute_module.l2_bandwidth_per_cycle
            )

            return total_cycle_count

    class L1TileSimulator:
        def __init__(
            self,
            M: int,
            N: int,
            K: int,
            data_type: DataType,
            mapping: "Matmul.Mapping",
            chiplet_module: Device,
            look_up_table: pd.DataFrame,
        ):
            # print(f'L1 tile: {M} {N} {K}')
            self.M = M
            self.N = N
            self.K = K
            self.compute_cycle_count = self.simulate_l1_tile_compute_cycle_count(
                M, N, K, data_type, mapping, chiplet_module, look_up_table
            )

        def simulate_l1_tile_compute_cycle_count(
            self,
            M: int,
            N: int,
            K: int,
            data_type: DataType,
            mapping: "Matmul.Mapping",
            chiplet_module: Device,
            look_up_table: pd.DataFrame,
        ):
            assert (
                M * K + K * N + M * N
                <= chiplet_module.compute_module.core.SRAM_size
                // data_type.word_size
                // 2
            )

            M_tiling_factor = mapping.l0_M_tiling_factor
            N_tiling_factor = mapping.l0_N_tiling_factor
            K_tiling_factor = mapping.l0_K_tiling_factor
            assert (
                M_tiling_factor * K_tiling_factor * N_tiling_factor
                <= chiplet_module.compute_module.core.systolic_array_count
            )

            compute_cycle_count = ceil(
                Matmul.simulate_systolic_array_cycle_count(
                    look_up_table,
                    ceil(M / M_tiling_factor),
                    ceil(N / N_tiling_factor),
                    ceil(K / K_tiling_factor),
                    chiplet_module.compute_module.core.systolic_array.array_height,
                    chiplet_module.compute_module.core.systolic_array.array_width,
                    chiplet_module.compute_module.core.systolic_array.mac_per_cycle,
                    mapping.dataflow,
                )
                + (K_tiling_factor - 1)
                * M
                * N
                / chiplet_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
            )

            return compute_cycle_count

    @staticmethod
    def simulate_systolic_array_cycle_count(
        look_up_table: pd.DataFrame,
        M,
        N,
        K,
        array_height,
        array_width,
        mac_per_clock,
        dataflow="os",
    ):
        print(f'start: {M} {N} {K} {array_height} {array_width} {mac_per_clock} {dataflow}')
        assert M * N * K * array_height * array_width * mac_per_clock != 0
        # if matrix size is large enough to fully utilize the systolic array
        if M >= array_height and N >= array_width:
            if (
                # SA utilization is larger than 128, the performance is close to theoretical performance
                M * N * K / array_height / array_width / max(array_height, array_width)
                >= 128
            ):
                return ceil(
                    M * N * K / array_height / array_width / mac_per_clock / 0.99
                )
            elif (
                # if between 64 - 128, still close with less efficiency factor
                M * N * K / array_height / array_width / max(array_height, array_width)
                >= 64
            ):
                return ceil(
                    M * N * K / array_height / array_width / mac_per_clock / 0.98
                )
        # SA width is under utilized but height is full filled
        # if threshold 64 is met, still modeled as theoretical performance
        elif M >= array_height and N < array_width:
            if K * M / array_height / max(array_height, array_width) >= 64:
                util_rate = N / array_width / 0.98
                return ceil(
                    M * N * K / array_height / array_width / mac_per_clock / util_rate
                )
        # similar as above, but height is under utilized
        elif M < array_height and N >= array_width:
            if K * N / array_width / max(array_height, array_width) >= 64:
                util_rate = M / array_height / 0.98
                return ceil(
                    M * N * K / array_height / array_width / mac_per_clock / util_rate
                )
        else:
            # both height and width under utilized
            assert M < array_height and N < array_width
            if K / max(array_height, array_width) >= 64:
                util_rate = M / array_height * N / array_width / 0.98
                return ceil(
                    M * N * K / array_height / array_width / mac_per_clock / util_rate
                )
            # if none of above fits, try the lookup table
        print('start look up table')
        try:
            cycle_count = look_up_table.loc[
                (M, N, K, array_height, array_width, dataflow), "cycle_count"
            ].item()
        except KeyError:
            try:
                cycle_count = look_up_table.loc[
                    (N, M, K, array_height, array_width, dataflow), "cycle_count"
                ].item()
            except KeyError:
                print('not found in look up table')
                config = f"./systolic_array_model/temp/systolic_array_{os.getpid()}.cfg"
                os.makedirs(os.path.dirname(config), exist_ok=True)
                with open(config, "w") as f:
                    f.writelines("[general]\n")
                    f.writelines("run_name = systolic_array\n\n")
                    f.writelines("[architecture_presets]\n")
                    f.writelines("ArrayHeight:    " + str(array_height) + "\n")
                    f.writelines("ArrayWidth:     " + str(array_width) + "\n")
                    f.writelines("IfmapSramSzkB:    " + str(1024) + "\n")
                    f.writelines("FilterSramSzkB:   " + str(1024) + "\n")
                    f.writelines("OfmapSramSzkB:    " + str(1024) + "\n")
                    f.writelines("IfmapOffset:    0\n")
                    f.writelines("FilterOffset:   10000000\n")
                    f.writelines("OfmapOffset:    20000000\n")
                    f.writelines("Dataflow : " + dataflow + "\n")
                    f.writelines("Bandwidth : " + "100" + "\n")
                    f.writelines("MemoryBanks: 1\n\n")
                    f.writelines("[run_presets]\n")
                    f.writelines("InterfaceBandwidth: CALC\n")

                topology = f"./systolic_array_model/temp/matmul_{os.getpid()}.csv"
                with open(topology, "w") as f:
                    f.writelines("Layer, M, N, K\n")
                    f.writelines(f"matmul1, {M}, {N}, {K},\n")

                logpath = f"./systolic_array_model/temp/"
                s = scalesim(
                    save_disk_space=True,
                    verbose=False,
                    config=config,
                    topology=topology,
                    input_type_gemm=True,
                )
                s.run_scale(top_path=logpath)

                cycle_count = s.runner.single_layer_sim_object_list[0].total_cycles
                util_rate = s.runner.single_layer_sim_object_list[0].overall_util
                with open(
                    f"./systolic_array_model/look_up_table_{array_height}_{array_width}.csv",
                    "a",
                ) as f:
                    f.writelines(
                        f"{M},{N},{K},{array_height},{array_width},{dataflow},{cycle_count},{util_rate:.3f}\n"
                    )
                look_up_table.loc[(M, N, K, array_height, array_width, dataflow), :] = [
                    cycle_count,
                    util_rate,
                ]
                if len(look_up_table) % 10 == 0:
                    look_up_table.sort_index(inplace=True)
                # look_up_table.to_csv(f"./systolic_array_model/look_up_table_{array_height}_{array_width}.csv")
                print(f"Appended to file: look_up_table_{array_height}_{array_width}.csv")
        # if (
        #     dataflow == "os"
        # ):  # scalesim assumes collecting output is not on critical path in os
        #     cycle_count += min(array_height, array_width, M, N)
        # if True:
        #     print(f"{M}x{N}x{K}x{array_height}x{array_width}x{dataflow}: {cycle_count}")
        # new_table = look_up_table[~look_up_table.index.duplicated(keep='first')]
        # if look_up_table.shape[0]-new_table.shape[0]>=1:
        #     print(look_up_table)
        #     print(look_up_table.duplicated(keep=False))
        #     exit()
        # print(f'end: {M} {N} {K} {array_height} {array_width} {mac_per_clock} {dataflow}')
        # assert isinstance(cycle_count, float), f"cycle_count: {cycle_count}"
        return ceil(cycle_count / mac_per_clock)

    def run_on_gpu(
        self,
    ):
        # import subprocess
        # subprocess.run(['nvidia-smi', '-q', '–d', 'CLOCK'])
        input1 = torch.randn(
            self.computational_graph.M,
            self.computational_graph.K,
            dtype=torch.bfloat16,
            device="cuda:0",
        )
        input2 = torch.randn(
            self.computational_graph.K,
            self.computational_graph.N,
            dtype=torch.bfloat16,
            device="cuda:0",
        )
        latencies = []
        input1_dummy = torch.ones(4096, 4096).cuda()
        input2_dummy = torch.ones(4096, 4096).cuda()
        # warmup
        for _ in range(3):
            torch.matmul(input1_dummy, input2_dummy)
            torch.cuda.synchronize()
            time.sleep(1)
        for _ in range(self.iterations):
            # x = torch.matmul(input1_dummy, input2_dummy)  # flush the cache
            # torch.cuda.synchronize()
            start = time.time()
            output = torch.matmul(input1, input2)
            torch.cuda.synchronize()
            end = time.time()
            assert list(output.shape) == [
                self.computational_graph.M,
                self.computational_graph.N,
            ]
            latencies.append(end - start)
            # time.sleep(1)

        self.latency_on_gpu = (
            statistics.median(latencies)
            # min(latencies)
            # - self.gpu_kernel_launch_overhead()
            # - 4e-5
            # min(latencies) - 8e-6
        )  # GPU launch kernel overhead and PyTorch overhead
        return self.latency_on_gpu

    @staticmethod
    def gpu_kernel_launch_overhead():
        size = 1
        latencies = []
        for _ in range(50):
            a = torch.randn(size, size, device="cuda")
            b = torch.randn(size, size, device="cuda")
            torch.cuda.synchronize()
            start = time.time()
            c = torch.matmul(a, b)
            torch.cuda.synchronize()
            end = time.time()
            latencies.append(end - start)
        avg_overhead = statistics.median(latencies)
        print("GPU kernel launch overhead: ", avg_overhead * 1e3, "ms")
        # print(latencies)
        return avg_overhead

