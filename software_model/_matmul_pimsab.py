from hardware_model.device import Device
from pimsab_exp.run_pimsab_gemm import run_gemm
from pimsab_kernel.gemm import gemm_tiled_compute
from math import ceil, log2, floor
from software_model.utils import Tensor, DataType, simdram_op_latency_dict
import numpy as np

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
    elif compile_mode == "heuristic-PIMSAB-sim-v3":
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
        print(f"precision: {input_acc}bit")
        accumulate_acc = input_acc+16
        debug = False
        tile_compute_latency ,_ = gemm_tiled_compute(pcb_module.compute_module,M_tile,K_tile,N_tile,input_acc,accumulate_acc)
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
            tile_RKN_compute_latency ,_ = gemm_tiled_compute(pcb_module.compute_module,M_remain,K_tile,N_tile,input_acc,accumulate_acc)
            tile_latency[-1, :N_t, :K_t] = tile_RKN_compute_latency
        if N_remain != 0:
            tile_MKR_compute_latency ,_ = gemm_tiled_compute(pcb_module.compute_module,M_tile,K_tile,N_remain,input_acc,accumulate_acc)
            tile_latency[:M_t, -1, :K_t] = tile_MKR_compute_latency
        if K_remain != 0:
            tile_MRN_compute_latency ,_ = gemm_tiled_compute(pcb_module.compute_module,M_tile,K_remain,N_tile,input_acc,accumulate_acc)
            tile_latency[:M_t, :N_t, -1] = tile_MRN_compute_latency
        if M_remain * N_remain != 0:
            tile_RKR_compute_latency ,_ = gemm_tiled_compute(pcb_module.compute_module,M_remain,K_tile,N_remain,input_acc,accumulate_acc)
            tile_latency[-1, -1, :K_t] = tile_RKR_compute_latency
        if M_remain * K_remain != 0:
            tile_RRN_compute_latency ,_ = gemm_tiled_compute(pcb_module.compute_module,M_remain,K_remain,N_tile,input_acc,accumulate_acc)
            tile_latency[-1, :N_t, -1] = tile_RRN_compute_latency
        if N_remain * K_remain != 0:
            tile_MRR_compute_latency ,_ = gemm_tiled_compute(pcb_module.compute_module,M_tile,K_remain,N_remain,input_acc,accumulate_acc)
            tile_latency[:M_t, -1, -1] = tile_MRR_compute_latency
        if M_remain * N_remain * K_remain != 0:
            tile_RRR_compute_latency ,_ = gemm_tiled_compute(pcb_module.compute_module,M_remain,K_remain,N_remain,input_acc,accumulate_acc)
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
