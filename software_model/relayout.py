from hardware_model.device import Device
from math import ceil
from software_model.utils import  simdram_op_latency_dict, simdram_PE_op_latency_dict
from software_model.utils import find_closest_divisor
from software_model.strategy import Strategy
from software_model.stats import Stats
from software_model.matmul import Matmul, BatchedMatmul

def has_same_characters(str1, str2):
    return sorted(list(str1)) == sorted(list(str2))

class Relayout:
    def __init__(self, device: Device,  matmul_1:Matmul, strategy_1: Strategy, matmul_2:Matmul, strategy_2: Strategy):
        self.device = device
        self.matmul_1 = matmul_1
        self.matmul_2 = matmul_2
        self.strategy_1 = strategy_1
        self.strategy_2 = strategy_2
        self.tile_mapping_1 = self.strategy_1.tile_mapping
        self.tile_mapping_2 = self.strategy_2.tile_mapping
        self.arr_mapping_1 = self.strategy_1.arr_mapping
        self.arr_mapping_2 = self.strategy_2.arr_mapping
        self.arr_mapping_1_M = "R" if 'M' in self.arr_mapping_1['R'] else "C"
        self.arr_mapping_1_K = "R" if 'K' in self.arr_mapping_1['R'] else "C"
        self.arr_mapping_1_N = "R" if 'N' in self.arr_mapping_1['R'] else "C"
        self.arr_mapping_2_M = "R" if 'M' in self.arr_mapping_2['R'] else "C"
        self.arr_mapping_2_K = "R" if 'K' in self.arr_mapping_2['R'] else "C"
        self.arr_mapping_2_N = "R" if 'N' in self.arr_mapping_2['R'] else "C"
        if not self.need_host():
            strategy_1.output_resident = True
            strategy_2.input_resident = True
            self.inplace_relayout_latency = self.get_inplace_relayout_latency()
        else:
            strategy_1.output_resident = False
            strategy_2.input_resident = False
            self.inplace_relayout_latency = 0

    def need_host(self):
        """
        Calculate the latency of the relayout operation.
        """
        
        # paired tile mapping
        if has_same_characters(self.tile_mapping_1['M'], self.tile_mapping_2['M']) and has_same_characters(self.tile_mapping_1['N'], self.tile_mapping_2['K']):
            #paired arr mapping
            if self.arr_mapping_1_M == self.arr_mapping_2_M and self.arr_mapping_1_N == self.arr_mapping_2_K and self.arr_mapping_1_K == self.arr_mapping_2_N:
                # No need to relayout
                return False
        #relayout by loading to host and write back
        return True
    
    def get_inplace_relayout_latency(self):
        assert self.need_host() == False
        self.tile_mapping_1['M']
        #extract duplication of MK matrix of kernel 2
        parallelisms = ['A', 'B', 'D']
        M_K_dup = []
        for c in parallelisms:
            # if not tiled along A/B/D
            # we need to duplicate this tile to corresponding place
            if c not in self.tile_mapping_2['M'] and c not in self.tile_mapping_2['K']:
                M_K_dup.append(c)
        array_broadcast_enable = False
        bank_broadcast_enable = False
        latency = 0
        if 'A' in self.strategy_2.broadcast:
            array_broadcast_enable = True
        if 'B' in self.strategy_2.broadcast:
            bank_broadcast_enable = True
        print(f"array_broadcast_enable: {array_broadcast_enable}, bank_broadcast_enable: {bank_broadcast_enable}")
        # TODO: What is the maximum available bandwidth here?
        latency = self.matmul_2.M* self.matmul_2.K * self.matmul_2.data_type.word_size / self.device.compute_module.bandwidth
        print(f"latency {latency} = {self.matmul_2.M} * {self.matmul_2.K} * {self.matmul_2.data_type.word_size} / {self.device.compute_module.bandwidth}")
        # print(f"latency {latency} = {tile_1} * {tile_2} * {word_size} / {pcb_module.compute_module.bandwidth}")
        if 'A' in M_K_dup and not array_broadcast_enable:
            # duplicate tile to every device
            latency *= self.device.compute_module.bank.arr_count
        if 'B' in M_K_dup and not bank_broadcast_enable:
            # duplicate tile to every bank
            latency *= self.device.compute_module.bank_count
        if 'D' in M_K_dup:
            # duplicate tile to every device
            latency *= self.device.compute_module.bank.device_count
        return latency