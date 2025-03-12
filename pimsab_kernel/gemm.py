import math
import hardware_model.compute_module_pimsab as cmp
def get_array_broadcast_cycles(precision):
    return precision + 1
def get_mul_cycles(precision):
    return precision*precision + 3 * precision - 2 + 1
def get_add_cycles(precision):
    return precision + 1 + 1
def get_array_reduce_cycles(precision, pimsab:cmp.ComputeModulePIMSAB):
    '''
    assumes pipelined reduction tree
    '''
    levels = math.ceil(math.log2(pimsab.tile.arr_count))
    return precision + levels + 1
def get_array_reduce_cycles_rowadd(precision, pimsab:cmp.ComputeModulePIMSAB):
    levels = math.ceil(math.log2(pimsab.tile.arr_count))
    cycles=0
    for i in range(1, levels+1):
        cycles+= precision + i
        distance = i+1 if i%2 else i
        cycles += distance
    return cycles
                        

def gemm_tiled_compute(pimsab:cmp.ComputeModulePIMSAB, M, K, N, precision_input, precision_accumulate):
    total_cycles = 0
    num_reqs = 0
    ncols = pimsab.tile.arr_cols
    narrays = pimsab.tile.arr_count
    ntiles = pimsab.tile_count
    adder = True
    for n in range(math.ceil(N/ncols)):
        for m in range(math.ceil(M/ntiles)):
            increase_precision_index = 1
            two_to_n = 1
            curr_iter = 0
            precision_accumulate_temp = precision_input*2
            mac_cycles = 0
            for k in range(math.ceil(K/narrays)):
                mac_cycles += get_array_broadcast_cycles(precision_input)
                num_reqs += 1
                mac_cycles += get_mul_cycles(precision_input)
                num_reqs += 1

                if curr_iter == increase_precision_index:
                    precision_accumulate_temp = min(precision_accumulate_temp+1, precision_accumulate)
                    increase_precision_index += two_to_n
                    two_to_n *=2
                curr_iter += 1

                mac_cycles += get_add_cycles(precision_accumulate_temp)
                num_reqs += 1
            
                # print(total_cycles)

            if adder: 
                m_iter_cycles = max(mac_cycles, get_array_reduce_cycles(precision_accumulate_temp, pimsab)) if m > 0  else mac_cycles
            else:
                m_iter_cycles = mac_cycles+ get_array_reduce_cycles_rowadd(precision_accumulate_temp, pimsab) if m > 0 else mac_cycles
            total_cycles += m_iter_cycles
        total_cycles += get_array_reduce_cycles(precision_accumulate_temp, pimsab) if adder else get_array_reduce_cycles_rowadd(precision_accumulate_temp, pimsab) #reduction of last m iteration
    latency = total_cycles/pimsab.clock_freq
    # print({pimsab.clock_freq})
    print(f"gemm_kernel_cycles: {total_cycles}")

    return latency, 0