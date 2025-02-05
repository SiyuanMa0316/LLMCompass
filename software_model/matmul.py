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


    def simdram_heuristic_tiling_v2(self, pcb_module: Device):
        
        def find_tile_K(row_limits : int, grow : int = 32) -> int:
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
        pcb_module.io_module.bandwidth = 19.6 * 8 # bandwidth in bits per ns
        M = self.computational_graph.M
        N = self.computational_graph.N
        K = self.computational_graph.K
        M_K_bits = M * K * self.data_type.word_size * 8
        K_N_bits = K * N * self.data_type.word_size * 8
        M_N_bits = M * N * self.data_type.word_size * 8
        # channels, bank, row size
        print(f"Heuristic-SIMDRAM Tiling Simulation: M {self.M}, K {self.K}, N {self.N}")
        """
        Hierarchy-Level: Rank -> device = bank -> array
        """
        col_per_array = 128
        row = 131072
        array_per_device_bank = 64
        bank = 16
        device = 8
        rank = 1
        capacity_per_array = col_per_array * row
        capacity_per_bank = capacity_per_array * array_per_device_bank
        capacity_per_device = capacity_per_bank * bank
        capacity_per_rank = capacity_per_device * device
        total_capacity = capacity_per_rank * rank

        print(f"Input GEMM Storage Size: Input + Output {(M_K_bits + K_N_bits + M_N_bits)/1024/1024/1024/8}GB")
        print(f"capacity_per_array:{capacity_per_array/1024/1024/8}MB\n"+ 
            f"Capacity_per_bank:{capacity_per_bank/1024/1024/8}MB\n"+ 
            f"Capacity_per_device:{capacity_per_device/1024/1024/1024/8}GB\n"+
            f"Capacity_per_rank:{capacity_per_rank/1024/1024/1024/8}GB\n"+
            f"Total_capacity:{total_capacity/1024/1024/1024/8}GB")
        
        
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
        n = 1
        col_per_array_pow = floor(log2(col_per_array))
        while n < col_per_array_pow:
            n = n + 1
        tile_N = 2 ** (n // 2)
        tile_M = 2 ** (n - n // 2)
        factor_N = ceil(N / tile_N)
        factor_M = ceil(M / tile_M)
        # find nearest power of 2 of tile_K that satisfy the row limit
        tile_K,accum_bits = find_tile_K(row)
        factor_K = floor(K / tile_K)
        remain_K = K - factor_K * tile_K
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

        print(f"A = {M} x {K} = {factor_M} x {factor_K} x ({tile_M} x {tile_K})")
        print(f"B = {K} x {N} = {factor_K} x {factor_N} x ({tile_K} x {tile_N})")
        print(f"Remain workloads {tile_M} x {remain_K} x {tile_N}")

        print(f"column utilization per array:{col_fragmentation_per_array * 100:.2f}%")
        print(f"row utilization per array:{row_fragmentation_per_array * 100:.2f}%")

        #per array latency
        add_latency_per_array = tile_K * simdram_op_latency_dict[self.data_type.name]['add']
        mul_latency_per_array = tile_K * simdram_op_latency_dict[self.data_type.name]['mul']
        compute_latency_per_array = add_latency_per_array + mul_latency_per_array

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
        
        _summary_array_level__: 
            Every array will have same bigger-tile B within same device,
            partition complete A into each array at batch size of 'array' along the M-dimension (downwards from A(1,1) to A(factor_M,1)).
            iterations in each array is determined by: factor_array = ceil(factor_M / array).
            
            __computations__: #adds = #muls = tile_k
            
            __output_bit_width__: output_bits_per_array = tile_M * tile_N * accum_bits
        
            __latency__: Each iteration within whole array has overlapped latency = 'compute_latency_per_array', 

        """
        factor_array = ceil(factor_M / array_per_device_bank)
        output_bits_per_array = col_per_array * accum_bits
        """
        __summary_device_level__: 
            Every device will have different bigger-tile B from assigned column vector,
            partition column vector of B into each device at batch size of 'device' along the K-dimension (downwards from B(1,1) to B(factor_K,1)). 
            iteration of each bank is determined by: factor_device = ceil(factor_K / device).
 
            __computations__: #adds = #muls = tile_K * factor_array
            
            __output_bit_width__: output_bits_per_device = output_bits_per_array * array

            __latency__: Each iteration within one device has the exactly overlapped same latency = 'compute_latency_per_array * factor_array'
 
        """
        factor_device = ceil(factor_K / device)
        computation_per_device = tile_K * factor_array
        output_bits_per_device = output_bits_per_array * array_per_device_bank
        """
        __summary__bank_level__: 
            Every bank will have different column vector of bigger-tile B,
            partition complete B into column vector at batch size of 'bank' along the N-dimension (rightwards from B(1,1) to B(1, factor_N)).
            iteration in each rank is determined by: factor_bank = ceil(factor_N / bank).
            
            __computations__: #adds = #muls = tile_k * factor_array * factor_device * factor_bank
            
            __output_bit_width__: output_bits_per_bank = output_bits_per_device * device   

            __latency__: Each iteration within one bank has the exactly overlapped same latency = 'compute_latency_per_array',
            Here, bank is the highest-level of micro-architecture in workload scheduling, the total latency is determined by the #iterations of array computes.
            total_compute_latency = 'computer_latency_per_array' * 'factor_array'
            bank bandwidth = 8bits / cycle
            data_write_back_latency =   output_bits_per_bank /bank bandwidth 

        """
        factor_bank = ceil(factor_N / bank)
        output_bits_per_bank = output_bits_per_device * device
        computation_per_bank = computation_per_device * device
        """    
            __summary_rank_level__:
            We assume only 1 rank for now, no parallelism across rank. However, we do have to consider the total data transfer latency
            and also if it is possible to schedule pipeline computation.

            __computations__: #adds = #muls = M * N * K
            __output_bit_width: output_bits_per_rank = output_bits_per_bank * bank * bank_factor
            __latency__: 
            Compute latency is all overlapped within each bank. But we have multiple iterations of bank scheduling
            total_compute_latency = compute_latency_per_array * factor_bank
            rank bandwidth = bank_bandwidth * bank /cycle
            total_data_write_back_latency = output_bits_per_rank / rank bandwidth
            
            total_latency = total_data_write_back_latency + total_compute_latency
        """
        total_computations = computation_per_device * computation_per_bank * bank
        total_output_bits = output_bits_per_bank * bank * factor_bank
        write_back_latency =  total_output_bits / pcb_module.io_module.bandwidth
        compute_latency = compute_latency_per_array * factor_array

        total_latency = compute_latency + write_back_latency
        # data amplification due to the vectorization of GEMM
        # each tile of B is duplicated factor_M x factor_N x factor_K times
        amplification_bits = factor_M * factor_N * factor_K * (tile_K * tile_N) * self.data_type.word_size * 8

        """
        We need to process the remaining workloads
        remain_A = factor_M x tile_M x remain_K , 
        remain_B = remain_K x tile_N x factor_N
        
        We still follow the same procedure to schedule the workload across the device and bank
        
        Based on previous calculation, the remaining smallest workload size (tile_M x remain_K, remain_K x tile_N)
        is guaranteed to fit within single array. 

        Since we only have 1 col/row vector left, no partition across bank device needed. 
        
        1. Scatter each remaining B tiles into arrays across the device: every array has same tile B but different tile A.
        remain_factor_array = ceil(factor_M / array)
        each array have overlapped latency and same total ops #adds = #muls = remain_K
        
        2. B tiles are partitioned across banks (following same procedure as before: partition N along bank)
        remain_factor_bank = ceil(factor_N / bank)
        each bank have overlapped latency and same total ops #adds = #muls = remain_K * array
        
        __remain_compute_latency__: remain_factor_array * remain_compute_latency_per_array
        __remain_bit_width__: same output bitwidth each array as normal schedule. 
        Total bitwidths output for every bank = (remain_array_factor * remain_bank_factor) * output_bits_per_array * array

        __remain_write_back_latency__: remain_factor_bank * remain_data_width /(bank_bandwidth)
        """
        remain_factor_array = ceil(factor_M / array_per_device_bank)
        remain_factor_bank = ceil(factor_N / bank)
        
        remain_add_latency_per_array = remain_K * simdram_op_latency_dict[self.data_type.name]['add']
        remain_mul_latency_per_array= remain_K * simdram_op_latency_dict[self.data_type.name]['mul']
        
        remain_compute_latency_per_array = (remain_add_latency_per_array  + remain_mul_latency_per_array)

        remain_compute_latency = remain_factor_array * remain_compute_latency_per_array
        remain_output_bits_per_bank = (remain_factor_array * remain_factor_bank) * output_bits_per_array * array_per_device_bank
        remain_write_back_latency = remain_output_bits_per_bank * bank / pcb_module.io_module.bandwidth

        remain_latency = remain_write_back_latency + remain_compute_latency

        amplification_bits = amplification_bits + self.data_type.word_size * 8 * (tile_N * remain_K) * factor_N * factor_M

        total_compute_latency = remain_compute_latency + compute_latency
        total_write_back_latency = remain_write_back_latency + write_back_latency

        total_latency = remain_latency + total_latency

        print(f"SIMDRAM Heuristic Tiling V2: total latency {total_latency} ns, compute_latency {compute_latency} ns, total_write_back_latency {total_write_back_latency} ns")
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


    def compile_and_simulate(
        self,
        pcb_module: Device,
        compile_mode: str = "exhaustive",
    ):
        if pcb_module.type=="systolic":
            return self.compile_and_simulate_systolic(pcb_module, compile_mode)
        elif pcb_module.type=="pimsab":
            return self.compile_and_simulate_pimsab(pcb_module, compile_mode)
        elif pcb_module.type == "simdram":
            return self.simdram_heuristic_tiling_v2(pcb_module)
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

