from software_model.operators import (
    Operator,
    Reshape,
    Concat,
    Transpose,
)
from software_model.matmul import Matmul, BatchedMatmul
from software_model.softmax import Softmax
from software_model.layernorm import LayerNorm
from software_model.gelu import GeLU
from software_model.relayout import Relayout


from software_model.utils import Tensor, DataType
from software_model.communication_primitives import AllReduceMultiPCB
from math import ceil
from typing import List
from hardware_model.system import System
import csv


def dump_log(logs: dict, name: str, e2e_latency, stats, overhead):
    logs[name] = {}
    logs[name]['latency'] = e2e_latency
    logs[name]['compute_latency'] = stats.compute_latency
    logs[name]['io_latency'] = stats.io_latency
    logs[name]['kernel_launch_overhead'] = overhead
    logs[name]['simd_utilization'] = stats.simd_utilization
    logs[name]['tiling_utilization'] = stats.tiling_utilization
    logs[name]['capacity_utilization'] = stats.capacity_utilization
    return logs

class TransformerBlockInitComputationTP(Operator):
    def __init__(self, d_model, n_heads, device_count, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.d_model = d_model
        self.n_heads = n_heads
        self.device_count = device_count
        # parameters per device
        d = d_model
        self.Wq = Tensor([d, d // device_count], data_type)
        self.Wk = Tensor([d, d // device_count], data_type)
        self.Wv = Tensor([d, d // device_count], data_type)
        self.W0 = Tensor([d // device_count, d], data_type)
        self.W1 = Tensor([d, 4 * d // device_count], data_type)
        self.W2 = Tensor([4 * d // device_count, d], data_type)
        # operators per device
        # # multi-head attention
        self.Q_proj = Matmul(data_type)
        self.K_proj = Matmul(data_type)
        self.V_proj = Matmul(data_type)
        self.Q_reshape = Reshape(data_type)
        self.K_reshape = Reshape(data_type)
        self.V_reshape = Reshape(data_type)
        self.Q_transpose = Transpose(data_type)
        self.K_transpose = Transpose(data_type)
        self.V_transpose = Transpose(data_type)
        self.Q_mul_K = BatchedMatmul(data_type)
        self.A_softmax = Softmax(data_type)
        self.A_mul_V = BatchedMatmul(data_type)
        self.H_transpose = Transpose(data_type)
        self.H_reshape = Reshape(data_type)
        self.H_matmul0 = Matmul(data_type)
        self.layer_norm0 = LayerNorm(data_type)
        self.allreduce_mha = AllReduceMultiPCB(data_type)
        # # feed-forward network
        self.H_matmul1 = Matmul(data_type)
        self.H_gelu = GeLU(data_type)
        self.H_matmul2 = Matmul(data_type)
        self.layer_norm1 = LayerNorm(data_type)
        self.allreduce_ffn = AllReduceMultiPCB(data_type)

    def __call__(self, X: Tensor) -> Tensor:
        # b: batch size
        # s: sequence length
        # d: hidden dimension
        # d_h: dimension per head
        b, s, d = X.shape
        assert d == self.d_model
        h = self.n_heads
        dev_cnt = self.device_count
        d_h = d // h

        # multi-head attention
        Q = self.Q_proj(X, self.Wq)  # [b, s, d / dev_cnt]
        assert Q.shape == [b, s, d // dev_cnt]
        K = self.K_proj(X, self.Wk)  # [b, s, d / dev_cnt]
        V = self.V_proj(X, self.Wv)  # [b, s, d / dev_cnt]
        Q = self.Q_reshape(Q, [b, s, h // dev_cnt, d_h])
        K = self.K_reshape(K, [b, s, h // dev_cnt, d_h])
        V = self.V_reshape(V, [b, s, h // dev_cnt, d_h])
        Q_T = self.Q_transpose(Q, [0, 2, 1, 3])  # [b, h / dev_cnt, s, d_h]
        assert Q_T.shape == [b, h // dev_cnt, s, d_h]
        K_T = self.K_transpose(K, [0, 2, 3, 1])  # [b, h / dev_cnt, d_h, s]
        assert K_T.shape == [b, h // dev_cnt, d_h, s]
        V_T = self.V_transpose(V, [0, 2, 1, 3])  # [b, h / dev_cnt, s, d_h]
        assert V_T.shape == [b, h // dev_cnt, s, d_h]
        A = self.Q_mul_K(Q_T, K_T)  # [b, h / dev_cnt, s, s]
        assert A.shape == [b, h // dev_cnt, s, s]
        A_prob = self.A_softmax(A)
        H = self.A_mul_V(A_prob, V_T)  #  [b, h / dev_cnt, s, d_h]
        assert H.shape == [b, h // dev_cnt, s, d_h]
        H = self.H_transpose(H, [0, 2, 1, 3])  #  [b, s, h / dev_cnt, d_h]
        assert H.shape == [b, s, h // dev_cnt, d_h]
        H = self.H_reshape(H, [b, s, d // dev_cnt])
        assert H.shape == [b, s, d // dev_cnt]
        H0 = self.H_matmul0(H, self.W0)  #  [b, s, d]
        assert H0.shape == [b, s, d]
        H0 = self.layer_norm0(H0)
        assert H0.shape == [b, s, d]
        if dev_cnt > 1:
            H0 = self.allreduce_mha(H0)

        # feed-forward network
        H1 = self.H_matmul1(H0, self.W1)  # [b, s, 4 * d / dev_cnt]
        assert H1.shape == [b, s, 4 * d // dev_cnt]
        H1 = self.H_gelu(H1)
        H2 = self.H_matmul2(H1, self.W2)  #  [b, s, d]
        assert H2.shape == [b, s, d]
        H2 = self.layer_norm1(H2)
        if dev_cnt > 1:
            H2 = self.allreduce_ffn(H2)

        assert H2.shape == [b, s, d]
        return H2

    def roofline_model(self, system: System):
        device = system.device
        interconnect = system.interconnect

        qkv_latency = 3 * (
            self.Q_proj.roofline_model(device) + device.compute_module.overhead.matmul
        )
        q_mul_k_latency = (
            self.Q_mul_K.roofline_model(device) + device.compute_module.overhead.matmul
        )
        a_mul_v_latency = (
            self.A_mul_V.roofline_model(device) + device.compute_module.overhead.matmul
        )
        h_matmul0_latency = (
            self.H_matmul0.roofline_model(device)
            + device.compute_module.overhead.matmul
        )
        h1_matmul1_latency = (
            self.H_matmul1.roofline_model(device)
            + device.compute_module.overhead.matmul
        )
        h2_matmul2_latency = (
            self.H_matmul2.roofline_model(device)
            + device.compute_module.overhead.matmul
        )

        matmul_total_latency = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + h1_matmul1_latency
            + h2_matmul2_latency
        )

        # normalization
        softmax_latency = (
            self.A_softmax.roofline_model(device)
            + device.compute_module.overhead.softmax
        )
        layernorm_latency = (
            self.layer_norm0.roofline_model(device)
            + device.compute_module.overhead.layernorm
        )

        normlization_total_latency = softmax_latency + layernorm_latency * 2

        # gelu
        gelu_latency = (
            self.H_gelu.roofline_model(device) + device.compute_module.overhead.gelu
        )

        # allreduce
        if self.device_count > 1:
            allreduce_latency = self.allreduce_mha.simulate(interconnect)
            allreduce_total_latency = allreduce_latency * 2
        else:
            allreduce_total_latency = 0
            allreduce_total_latency = 0

        # others

        # print
        print("Roofline breakdown:")
        print(
            f"{qkv_latency}\n{q_mul_k_latency}\n{a_mul_v_latency}\n{h_matmul0_latency}\n{h1_matmul1_latency}\n{h2_matmul2_latency}\n{softmax_latency}\n{layernorm_latency}\n{layernorm_latency}\n{gelu_latency}\n{allreduce_latency}\n{allreduce_latency}\n"
        )
        self.roofline_log = f"{qkv_latency}, {q_mul_k_latency}, {a_mul_v_latency}, {h_matmul0_latency}, {h1_matmul1_latency}, {h2_matmul2_latency}, {softmax_latency}, {layernorm_latency}, {layernorm_latency}, {gelu_latency}, {allreduce_latency}, {allreduce_latency}"
        print("total:")
        print(
            f"{matmul_total_latency}\n{normlization_total_latency}\n{gelu_latency}\n{allreduce_total_latency}\n"
        )
        self.roofline_latency = (
            matmul_total_latency
            + normlization_total_latency
            + gelu_latency
            + allreduce_total_latency
        )
        return self.roofline_latency
    
    def compile_and_simulate_simdram(self, system: System, compile_mode: str, debug: bool = False):
        device = system.device
        interconnect = system.interconnect
       
        if compile_mode == "specific":
            qkv_mapping = self.Q_proj.find_simdram_mapping(device)
            q_mul_k_mapping = self.Q_mul_K.matmul.find_simdram_mapping(device)
            a_mul_v_mapping = self.A_mul_V.matmul.find_simdram_mapping(device)
            h_matmul0_mapping = self.H_matmul0.find_simdram_mapping(device)
            h1_matmul1_mapping = self.H_matmul1.find_simdram_mapping(device)
            h2_matmul2_mapping = self.H_matmul2.find_simdram_mapping(device)
        else:
            qkv_mapping = None
            q_mul_k_mapping = None
            a_mul_v_mapping = None
            h_matmul0_mapping = None
            h1_matmul1_mapping = None
            h2_matmul2_mapping = None

        # matmul
        logs = {}
        launch_overhead = device.compute_module.overhead.matmul
        print(f"simulating qkv: ")
        qkv_latency = 3 * (
            self.Q_proj.compile_and_simulate(device, compile_mode=compile_mode, strategy=qkv_mapping, debug=debug)
            + launch_overhead
        )
        logs['qkv'] = {}
        logs['qkv']['latency'] = 3*qkv_latency
        logs['qkv']['compute_latency'] = 3*self.Q_proj.stats.compute_latency
        logs['qkv']['io_latency'] = 3*self.Q_proj.stats.io_latency
        logs['qkv']['kernel_launch_overhead'] = 3*launch_overhead

        logs['qkv']['simd_utilization'] = self.Q_proj.stats.simd_utilization
        logs['qkv']['tiling_utilization'] = self.Q_proj.stats.tiling_utilization
        logs['qkv']['capacity_utilization'] = self.Q_proj.stats.capacity_utilization
        print(f"qkv latency: {qkv_latency}, compute latency: {self.Q_proj.stats.compute_latency}, io overhead: {self.Q_proj.stats.io_latency}")

        print("simulating q_mul_k")
        q_mul_k_latency = (
            self.Q_mul_K.compile_and_simulate(device, compile_mode=compile_mode, strategy=q_mul_k_mapping, debug=debug)
            + launch_overhead
        )
        # logs['q_mul_k'] = {}
        # logs['q_mul_k']['latency'] = q_mul_k_latency
        # logs['q_mul_k']['compute_latency'] = self.Q_mul_K.stats.compute_latency
        # logs['q_mul_k']['io_latency'] = self.Q_mul_K.stats.io_latency
        # logs['q_mul_k']['kernel_launch_overhead'] = launch_overhead
        print(f"q_mul_k latency: {q_mul_k_latency}, compute latency: {self.Q_mul_K.stats.compute_latency}, io overhead: {self.Q_mul_K.stats.io_latency}")

        logs = dump_log(logs, 'q_mul_k', q_mul_k_latency, self.Q_mul_K.stats, launch_overhead)

        print("simulating a_mul_v")
        a_mul_v_latency = (
            self.A_mul_V.compile_and_simulate(device, compile_mode=compile_mode, strategy=a_mul_v_mapping, debug=debug)
            + launch_overhead
        )
        # logs['a_mul_v'] = {}
        # logs['a_mul_v']['latency'] = a_mul_v_latency
        # logs['a_mul_v']['compute_latency'] = self.A_mul_V.stats.compute_latency
        # logs['a_mul_v']['io_latency'] = self.A_mul_V.stats.io_latency
        # logs['a_mul_v']['kernel_launch_overhead'] = launch_overhead

        logs = dump_log(logs, 'a_mul_v', a_mul_v_latency, self.A_mul_V.stats, launch_overhead)
        print(f"a_mul_v latency: {a_mul_v_latency}, compute latency: {self.A_mul_V.stats.compute_latency}, io overhead: {self.A_mul_V.stats.io_latency}")

        print("simulating h_matmul0")
        h_matmul0_latency = (
            self.H_matmul0.compile_and_simulate(device, compile_mode=compile_mode, strategy=h_matmul0_mapping, debug=debug)
            + launch_overhead
        )
        # logs['h_matmul0'] = {}
        # logs['h_matmul0']['latency'] = h_matmul0_latency
        # logs['h_matmul0']['compute_latency'] = self.H_matmul0.stats.compute_latency
        # logs['h_matmul0']['io_latency'] = self.H_matmul0.stats.io_latency
        # logs['h_matmul0']['kernel_launch_overhead'] = launch_overhead

        logs = dump_log(logs, 'h_matmul0', h_matmul0_latency, self.H_matmul0.stats, launch_overhead)
        print(f"h_matmul0 latency: {h_matmul0_latency}, compute latency: {self.H_matmul0.stats.compute_latency}, io overhead: {self.H_matmul0.stats.io_latency}")

        print("simulating h1_matmul1")
        h1_matmul1_latency = (
            self.H_matmul1.compile_and_simulate(device, compile_mode=compile_mode, strategy=h1_matmul1_mapping, debug=debug)
            + launch_overhead
        )
        # logs['h1_matmul1'] = {}
        # logs['h1_matmul1']['latency'] = h1_matmul1_latency
        # logs['h1_matmul1']['compute_latency'] = self.H_matmul1.stats.compute_latency
        # logs['h1_matmul1']['io_latency'] = self.H_matmul1.stats.io_latency
        # logs['h1_matmul1']['kernel_launch_overhead'] = launch_overhead

        
        logs = dump_log(logs, 'h1_matmul1', h1_matmul1_latency, self.H_matmul1.stats, launch_overhead)

        print(f"h1_matmul1 latency: {h1_matmul1_latency}, compute latency: {self.H_matmul1.stats.compute_latency}, io overhead: {self.H_matmul1.stats.io_latency}")

        print("simulating h2_matmul2")
        h2_matmul2_latency = (
            self.H_matmul2.compile_and_simulate(device, compile_mode=compile_mode, strategy=h2_matmul2_mapping, debug=debug)
            + launch_overhead
        )
        # logs['h2_matmul2'] = {}
        # logs['h2_matmul2']['latency'] = h2_matmul2_latency
        # logs['h2_matmul2']['compute_latency'] = self.H_matmul2.stats.compute_latency
        # logs['h2_matmul2']['io_latency'] = self.H_matmul2.stats.io_latency
        # logs['h2_matmul2']['kernel_launch_overhead'] = launch_overhead
        
        logs = dump_log(logs, 'h2_matmul2', h2_matmul2_latency, self.H_matmul2.stats, launch_overhead)
        print(f"h2_matmul2 latency: {h2_matmul2_latency}, compute latency: {self.H_matmul2.stats.compute_latency}, io overhead: {self.H_matmul2.stats.io_latency}")
        print("finish matmul simulation")

        matmul_total_latency = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + h1_matmul1_latency
            + h2_matmul2_latency
        )

       

        print(f"matmul total overhead: {8*launch_overhead}")
        print(f"matmul total latency w/o overhead: {matmul_total_latency - 8*launch_overhead}")
        print(f"matmul total latency: {matmul_total_latency}")

        # normalization
        softmax_latency = (
            self.A_softmax.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.softmax
        )
        layernorm_latency = (
            self.layer_norm0.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.layernorm
        )

        normlization_total_latency = softmax_latency + layernorm_latency * 2

        # gelu
        gelu_latency = (
            self.H_gelu.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.gelu
        )

        # allreduce
        if self.device_count > 1:
            allreduce_latency = self.allreduce_mha.simulate(interconnect)
            allreduce_total_latency = allreduce_latency * 2
        else:
            allreduce_latency = 0
            allreduce_total_latency = 0

        #log latencies to a csv file
        with open('transformer_prefill_profile.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Entry',
                'Q_K_V', 'Q_mul_k', 'A_mul_V', 
                'Wo_proj', 'W1_proj', 'W2_proj', 
                'Softmax', 'Layernorm', 
                'GeLU', 'Allreduce'
            ])
            # writer.writerow(['End2End Latency',
            #     qkv_latency, q_mul_k_latency, a_mul_v_latency, 
            #     h_matmul0_latency, h1_matmul1_latency, h2_matmul2_latency, 
            #     softmax_latency, layernorm_latency,
            #     gelu_latency, allreduce_latency
            # ])

            keys = ['entry','qkv', 'q_mul_k', 'a_mul_v', 'h_matmul0', 'h1_matmul1', 'h2_matmul2']

            logs['entry'] = {'latency': 'latency', 'compute_latency': 'compute_latency', 'io_latency': 'io_latency', 
                             'kernel_launch_overhead': 'kernel_launch_overhead', 'tiling_utilization': 'tiling_utilization',
                             'simd_utilization': 'simd_utilization', 'capacity_utilization': 'capacity_utilization'}
            

            latency_row = [logs[key]['latency'] for key in keys]
            compute_latency_row = [logs[key]['compute_latency'] for key in keys]
            io_latency_row = [logs[key]['io_latency'] for key in keys]
            launch_overhead_row = [logs[key]['kernel_launch_overhead'] for key in keys]
            simd_utilization_row = [logs[key]['simd_utilization'] for key in keys]
            capacity_utilization_row = [logs[key]['capacity_utilization'] for key in keys]
            tiling_utilization_row = [logs[key]['tiling_utilization'] for key in keys]

            writer.writerow(latency_row)
            writer.writerow(compute_latency_row)
            writer.writerow(io_latency_row)
            writer.writerow(launch_overhead_row)
            writer.writerow(simd_utilization_row)
            writer.writerow(capacity_utilization_row)
            writer.writerow(tiling_utilization_row)


        # others

        # print
        # print("breakdown:")
        # print(
        #     f"{qkv_latency}\n{q_mul_k_latency}\n{a_mul_v_latency}\n{h_matmul0_latency}\n{h1_matmul1_latency}\n{h2_matmul2_latency}\n{softmax_latency}\n{layernorm_latency}\n{layernorm_latency}\n{gelu_latency}\n{allreduce_latency}\n{allreduce_latency}\n"
        # )
        # print("total:")
        # print(
        #     f"{matmul_total_latency}\n{normlization_total_latency}\n{gelu_latency}\n{allreduce_total_latency}\n"
        # )
        self.latency = (
            matmul_total_latency
            + normlization_total_latency
            + gelu_latency
            + allreduce_total_latency
        )
        self.simulate_log = f"{qkv_latency}, {q_mul_k_latency}, {a_mul_v_latency}, {h_matmul0_latency}, {h1_matmul1_latency}, {h2_matmul2_latency}, {softmax_latency}, {layernorm_latency}, {layernorm_latency}, {gelu_latency}, {allreduce_latency}, {allreduce_latency}"
        return self.latency

    def compile_and_simulate_gpu(self, system: System, compile_mode: str):
        device = system.device
        interconnect = system.interconnect
        # matmul
        print(f"simulating qkv: ")
        qkv_latency = 3 * (
            self.Q_proj.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.matmul
        )

        print("simulating q_mul_k")
        q_mul_k_latency = (
            self.Q_mul_K.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.matmul
        )
        print("simulating a_mul_v")
        a_mul_v_latency = (
            self.A_mul_V.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.matmul
        )
        print("simulating h_matmul0")
        h_matmul0_latency = (
            self.H_matmul0.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.matmul
        )
        print("simulating h1_matmul1")
        h1_matmul1_latency = (
            self.H_matmul1.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.matmul
        )
        print("simulating h2_matmul2")
        h2_matmul2_latency = (
            self.H_matmul2.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.matmul
        )
        print("finish matmul simulation")

        matmul_total_latency = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + h1_matmul1_latency
            + h2_matmul2_latency
        )

       

        print(f"matmul total overhead: {8*device.compute_module.overhead.matmul}")
        print(f"matmul total latency w/o overhead: {matmul_total_latency - 8*device.compute_module.overhead.matmul}")
        print(f"matmul total latency: {matmul_total_latency}")

        # normalization
        softmax_latency = (
            self.A_softmax.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.softmax
        )
        layernorm_latency = (
            self.layer_norm0.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.layernorm
        )

        normlization_total_latency = softmax_latency + layernorm_latency * 2

        # gelu
        gelu_latency = (
            self.H_gelu.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.gelu
        )

        # allreduce
        if self.device_count > 1:
            allreduce_latency = self.allreduce_mha.simulate(interconnect)
            allreduce_total_latency = allreduce_latency * 2
        else:
            allreduce_latency = 0
            allreduce_total_latency = 0

        #log latencies to a csv file
        with open('transformer_latency_log.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                'Q_K_V', 'Q_mul_k', 'A_mul_V', 
                'Wo_proj', 'W1_proj', 'W2_proj', 
                'Softmax', 'Layernorm', 
                'GeLU', 'Allreduce'
            ])
            writer.writerow([
                qkv_latency, q_mul_k_latency, a_mul_v_latency, 
                h_matmul0_latency, h1_matmul1_latency, h2_matmul2_latency, 
                softmax_latency, layernorm_latency,
                gelu_latency, allreduce_latency
            ])
        # others

        # print
        # print("breakdown:")
        # print(
        #     f"{qkv_latency}\n{q_mul_k_latency}\n{a_mul_v_latency}\n{h_matmul0_latency}\n{h1_matmul1_latency}\n{h2_matmul2_latency}\n{softmax_latency}\n{layernorm_latency}\n{layernorm_latency}\n{gelu_latency}\n{allreduce_latency}\n{allreduce_latency}\n"
        # )
        # print("total:")
        # print(
        #     f"{matmul_total_latency}\n{normlization_total_latency}\n{gelu_latency}\n{allreduce_total_latency}\n"
        # )
        self.latency = (
            matmul_total_latency
            + normlization_total_latency
            + gelu_latency
            + allreduce_total_latency
        )
        self.simulate_log = f"{qkv_latency}, {q_mul_k_latency}, {a_mul_v_latency}, {h_matmul0_latency}, {h1_matmul1_latency}, {h2_matmul2_latency}, {softmax_latency}, {layernorm_latency}, {layernorm_latency}, {gelu_latency}, {allreduce_latency}, {allreduce_latency}"
        return self.latency
    def compile_and_simulate(self, system: System, compile_mode: str = "exhaustive", debug: bool = False):
        device = system.device
        if device.type == "simdram":
            return self.compile_and_simulate_simdram(system, compile_mode, debug)
        else:
            return self.compile_and_simulate_gpu(system, compile_mode)

    def run_on_gpu(self):
        # matmul
        qkv_latency = (
            self.Q_proj.run_on_gpu()  # - self.Q_proj.gpu_kernel_launch_overhead()
        ) * 3
        q_mul_k_latency = (
            self.Q_mul_K.run_on_gpu()  # - self.Q_mul_K.gpu_kernel_launch_overhead()
        )
        a_mul_v_latency = (
            self.A_mul_V.run_on_gpu()  # - self.A_mul_V.gpu_kernel_launch_overhead()
        )
        h_matmul0_latency = (
            self.H_matmul0.run_on_gpu()  # - self.H_matmul0.gpu_kernel_launch_overhead()
        )
        h1_matmul1_latency = (
            self.H_matmul1.run_on_gpu()  # - self.H_matmul1.gpu_kernel_launch_overhead()
        )
        h2_matmul2_latency = (
            self.H_matmul2.run_on_gpu()  # - self.H_matmul2.gpu_kernel_launch_overhead()
        )

        matmul_total_latency = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + h1_matmul1_latency
            + h2_matmul2_latency
        )

        # normalization
        softmax_latency = (
            self.A_softmax.run_on_gpu()  # - self.A_softmax.gpu_kernel_launch_overhead()
        )
        layernorm_latency = (
            self.layer_norm0.run_on_gpu()
            - self.layer_norm0.gpu_kernel_launch_overhead()
        )

        normlization_total_latency = softmax_latency + layernorm_latency * 2

        # gelu
        gelu_latency = (
            self.H_gelu.run_on_gpu()  # - self.H_gelu.gpu_kernel_launch_overhead()
        )

        # allreduce
        allreduce_total_latency = 0

        # others

        # print
        print("breakdown:")
        print(
            f"{qkv_latency}\n{q_mul_k_latency}\n{a_mul_v_latency}\n{h_matmul0_latency}\n{h1_matmul1_latency}\n{h2_matmul2_latency}\n{softmax_latency}\n{layernorm_latency}\n{layernorm_latency}\n{gelu_latency}\n"
        )
        print("total:")
        print(
            f"{matmul_total_latency}\n{normlization_total_latency}\n{gelu_latency}\n{allreduce_total_latency}\n"
        )
        self.latency_on_gpu = (
            matmul_total_latency
            + normlization_total_latency
            + gelu_latency
            + allreduce_total_latency
        )
        return self.latency_on_gpu


class TransformerBlockAutoRegressionTP(Operator):
    def __init__(self, d_model, n_heads, device_count, data_type: DataType, core_count_per_block=1):
        super().__init__(0, 0, 0, 0, data_type)
        self.d_model = d_model
        self.n_heads = n_heads
        self.device_count = device_count
        self.core_count_per_block=1
        # parameters per device
        d = d_model
        self.Wq = Tensor([d, d // device_count], data_type)
        self.Wk = Tensor([d, d // device_count], data_type)
        self.Wv = Tensor([d, d // device_count], data_type)
        self.W0 = Tensor([d // device_count, d], data_type)
        self.W1 = Tensor([d, 4 * d // device_count], data_type)
        self.W2 = Tensor([4 * d // device_count, d], data_type)
        # operators per device
        # # multi-head attention
        self.Q_proj = Matmul(data_type)
        self.K_proj = Matmul(data_type)
        self.V_proj = Matmul(data_type)
        self.Q_reshape = Reshape(data_type)
        self.K_reshape = Reshape(data_type)
        self.V_reshape = Reshape(data_type)
        self.Q_transpose = Transpose(data_type)
        self.K_transpose = Transpose(data_type)
        self.V_transpose = Transpose(data_type)
        self.K_concat = Concat(data_type)
        self.V_concat = Concat(data_type)
        self.Q_mul_K = BatchedMatmul(data_type)
        self.A_softmax = Softmax(data_type)
        self.A_mul_V = BatchedMatmul(data_type)
        self.H_transpose = Transpose(data_type)
        self.H_reshape = Reshape(data_type)
        self.H_matmul0 = Matmul(data_type)
        self.layer_norm0 = LayerNorm(data_type)
        self.allreduce_mha = AllReduceMultiPCB(data_type)
        # # feed-forward network
        self.H_matmul1 = Matmul(data_type)
        self.H_gelu = GeLU(data_type)
        self.H_matmul2 = Matmul(data_type)
        self.layer_norm1 = LayerNorm(data_type)
        self.allreduce_ffn = AllReduceMultiPCB(data_type)

    def __call__(self, x: Tensor, seq_len: int) -> Tensor:
        # b: batch size
        # s: sequence length
        # d: hidden dimension
        # d_h: dimension per head
        b, _, d = x.shape
        assert d == self.d_model
        s = seq_len
        h = self.n_heads
        dev_cnt = self.device_count
        d_h = d // h

        # KV cache
        K_cache = Tensor([b, h // dev_cnt, d_h, s], self.data_type)
        V_cache = Tensor([b, h // dev_cnt, s, d_h], self.data_type)

        # multi-head attention
        q = self.Q_proj(x, self.Wq)  # [b, 1, d / dev_cnt]
        assert q.shape == [b, 1, d // dev_cnt]
        k = self.K_proj(x, self.Wk)  # [b, 1, d / dev_cnt]
        v = self.V_proj(x, self.Wv)  # [b, 1, d / dev_cnt]
        q = self.Q_reshape(q, [b, 1, h // dev_cnt, d_h])
        k = self.K_reshape(k, [b, 1, h // dev_cnt, d_h])
        v = self.V_reshape(v, [b, 1, h // dev_cnt, d_h])
        q_T = self.Q_transpose(q, [0, 2, 1, 3])  # [b, h / dev_cnt, 1, d_h]
        assert q_T.shape == [b, h // dev_cnt, 1, d_h]
        k_T = self.K_transpose(k, [0, 2, 3, 1])  # [b, h / dev_cnt, d_h, 1]
        assert k_T.shape == [b, h // dev_cnt, d_h, 1]
        v_T = self.V_transpose(v, [0, 2, 1, 3])  # [b, h / dev_cnt, 1, d_h]
        assert v_T.shape == [b, h // dev_cnt, 1, d_h]
        K_T = self.K_concat(K_cache, k_T, 3)  # [b, h / dev_cnt, d_h, s+1]
        assert K_T.shape == [b, h // dev_cnt, d_h, s + 1]
        V_T = self.V_concat(V_cache, v_T, 2)  # [b, h / dev_cnt, s+1, d_h]
        assert V_T.shape == [b, h // dev_cnt, s + 1, d_h]
        a = self.Q_mul_K(q_T, K_T)  # [b, h / dev_cnt, 1, s+1]
        assert a.shape == [b, h // dev_cnt, 1, s + 1]
        a_prob = self.A_softmax(a)
        h0 = self.A_mul_V(a_prob, V_T)  #  [b, h / dev_cnt, 1, d_h]
        assert h0.shape == [b, h // dev_cnt, 1, d_h]
        h0 = self.H_transpose(h0, [0, 2, 1, 3])  #  [b, 1, h / dev_cnt, d_h]
        assert h0.shape == [b, 1, h // dev_cnt, d_h]
        h0 = self.H_reshape(h0, [b, 1, d // dev_cnt])
        assert h0.shape == [b, 1, d // dev_cnt]
        h0 = self.H_matmul0(h0, self.W0)  #  [b, 1, d]
        assert h0.shape == [b, 1, d]
        h0 = self.layer_norm0(h0)
        assert h0.shape == [b, 1, d]
        if dev_cnt > 1:
            h0 = self.allreduce_mha(h0)

        # feed-forward network
        h1 = self.H_matmul1(h0, self.W1)  # [b, 1, 4 * d / dev_cnt]
        assert h1.shape == [b, 1, 4 * d // dev_cnt]
        h1 = self.H_gelu(h1)
        h2 = self.H_matmul2(h1, self.W2)  #  [b, 1, d]
        assert h2.shape == [b, 1, d]
        h2 = self.layer_norm1(h2)
        if dev_cnt > 1:
            h2 = self.allreduce_ffn(h2)

        assert h2.shape == [b, 1, d]
        self.memory_requirement = (
            self.Wq.size * self.Wq.data_type.word_size
            + self.Wk.size * self.Wk.data_type.word_size
            + self.Wv.size * self.Wv.data_type.word_size
            + self.W0.size * self.W0.data_type.word_size
            + self.W1.size * self.W1.data_type.word_size
            + self.W2.size * self.W2.data_type.word_size
            + K_cache.size * K_cache.data_type.word_size
            + V_cache.size * V_cache.data_type.word_size
        )
        return h2

    def roofline_model(self, system: System):
        device = system.device
        interconnect = system.interconnect

        qkv_latency = 3 * (
            self.Q_proj.roofline_model(device) + device.compute_module.overhead.matmul
        )
        q_mul_k_latency = (
            self.Q_mul_K.roofline_model(device) + device.compute_module.overhead.matmul
        )
        a_mul_v_latency = (
            self.A_mul_V.roofline_model(device) + device.compute_module.overhead.matmul
        )
        h_matmul0_latency = (
            self.H_matmul0.roofline_model(device)
            + device.compute_module.overhead.matmul
        )
        h1_matmul1_latency = (
            self.H_matmul1.roofline_model(device)
            + device.compute_module.overhead.matmul
        )
        h2_matmul2_latency = (
            self.H_matmul2.roofline_model(device)
            + device.compute_module.overhead.matmul
        )

        matmul_total_latency = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + h1_matmul1_latency
            + h2_matmul2_latency
        )

        # normalization
        softmax_latency = (
            self.A_softmax.roofline_model(device)
            + device.compute_module.overhead.softmax
        )
        layernorm_latency = (
            self.layer_norm0.roofline_model(device)
            + device.compute_module.overhead.layernorm
        )

        normlization_total_latency = softmax_latency + layernorm_latency * 2

        # gelu
        gelu_latency = (
            self.H_gelu.roofline_model(device) + device.compute_module.overhead.gelu
        )

        # allreduce
        if self.device_count > 1:
            allreduce_latency = self.allreduce_mha.simulate(interconnect)
            allreduce_total_latency = allreduce_latency * 2
        else:
            allreduce_latency = 0
            allreduce_total_latency = 0

        # others

        # print
        print("Roofline breakdown:")
        print(
            f"{qkv_latency}\n{q_mul_k_latency}\n{a_mul_v_latency}\n{h_matmul0_latency}\n{h1_matmul1_latency}\n{h2_matmul2_latency}\n{softmax_latency}\n{layernorm_latency}\n{layernorm_latency}\n{gelu_latency}\n{allreduce_latency}\n{allreduce_latency}\n"
        )
        print("total:")
        print(
            f"{matmul_total_latency}\n{normlization_total_latency}\n{gelu_latency}\n{allreduce_total_latency}\n"
        )
        self.roofline_latency = (
            matmul_total_latency
            + normlization_total_latency
            + gelu_latency
            + allreduce_total_latency
        )
        # print(f'memory requirement: {self.memory_requirement/1e9*96}GB')
        self.roofline_log = f"{qkv_latency}, {q_mul_k_latency}, {a_mul_v_latency}, {h_matmul0_latency}, {h1_matmul1_latency}, {h2_matmul2_latency}, {softmax_latency}, {layernorm_latency}, {layernorm_latency}, {gelu_latency}, {allreduce_latency}, {allreduce_latency}"
        return self.roofline_latency

    def compile_and_simulate_simdram(self, system: System, compile_mode: str):
        device = system.device
        interconnect = system.interconnect
        # matmul
        print(f"simulating qkv: ")
        if compile_mode == "specific":
            qkv_mapping = self.Q_proj.find_simdram_mapping(device)
            print(f"qkv mapping: {qkv_mapping}")
            q_mul_k_mapping = self.Q_mul_K.matmul.find_simdram_mapping(device)
            print(f"q_mul_k mapping: {q_mul_k_mapping}")
            a_mul_v_mapping = self.A_mul_V.matmul.find_simdram_mapping(device)
            print(f"a_mul_v mapping: {a_mul_v_mapping}")
            h_matmul0_mapping = self.H_matmul0.find_simdram_mapping(device)
            print(f"h_matmul0 mapping: {h_matmul0_mapping}")
            h1_matmul1_mapping = self.H_matmul1.find_simdram_mapping(device)
            print(f"h1_matmul1 mapping: {h1_matmul1_mapping}")
            h2_matmul2_mapping = self.H_matmul2.find_simdram_mapping(device)
            print(f"h2_matmul2 mapping: {h2_matmul2_mapping}")
        # elif compile_mode == "heuristic":

        else:
            qkv_mapping = None
            q_mul_k_mapping = None
            a_mul_v_mapping = None
            h_matmul0_mapping = None
            h1_matmul1_mapping = None
            h2_matmul2_mapping = None
        #create  logs
        logs = {}
        launch_overhead = device.compute_module.overhead.matmul
        qkv_latency = 3 * (
            self.Q_proj.compile_and_simulate(device, compile_mode=compile_mode, strategy=qkv_mapping)
            + device.compute_module.overhead.matmul
        )
        logs['qkv'] = {}
        logs['qkv']['latency'] = 3*qkv_latency
        logs['qkv']['compute_latency'] = 3*self.Q_proj.stats.compute_latency
        logs['qkv']['io_latency'] = 3*self.Q_proj.stats.io_latency
        logs['qkv']['kernel_launch_overhead'] = 3*device.compute_module.overhead.matmul
        logs['qkv']['simd_utilization'] = self.Q_proj.stats.simd_utilization
        logs['qkv']['tiling_utilization'] = self.Q_proj.stats.tiling_utilization
        logs['qkv']['capacity_utilization'] = self.Q_proj.stats.capacity_utilization


        print(f"qkv latency: {qkv_latency}, compute latency: {3*self.Q_proj.stats.compute_latency}, io overhead: {3*self.Q_proj.stats.io_latency}, kernel launch overhead: {3*device.compute_module.overhead.matmul}")

        print("simulating q_mul_k")
        q_mul_k_latency = (
            self.Q_mul_K.compile_and_simulate(device, compile_mode=compile_mode, strategy=q_mul_k_mapping, debug=False)
            + device.compute_module.overhead.matmul
        )
        # logs['q_mul_k'] = {}
        # logs['q_mul_k']['latency'] = q_mul_k_latency
        # logs['q_mul_k']['compute_latency'] = self.Q_mul_K.stats.compute_latency
        # logs['q_mul_k']['io_latency'] = self.Q_mul_K.stats.io_latency
        # logs['q_mul_k']['kernel_launch_overhead'] = device.compute_module.overhead.matmul


        logs = dump_log(logs, 'q_mul_k', q_mul_k_latency, self.Q_mul_K.stats, launch_overhead)

        print(f"q_mul_k latency: {q_mul_k_latency}, compute latency: {self.Q_mul_K.stats.compute_latency}, io overhead: {self.Q_mul_K.stats.io_latency}, kernel launch overhead: {device.compute_module.overhead.matmul}")

        print("simulating a_mul_v")
        a_mul_v_latency = (
            self.A_mul_V.compile_and_simulate(device, compile_mode=compile_mode, strategy=a_mul_v_mapping)
            + device.compute_module.overhead.matmul
        )
        # logs['a_mul_v'] = {}
        # logs['a_mul_v']['latency'] = a_mul_v_latency
        # logs['a_mul_v']['compute_latency'] = self.A_mul_V.stats.compute_latency
        # logs['a_mul_v']['io_latency'] = self.A_mul_V.stats.io_latency
        # logs['a_mul_v']['kernel_launch_overhead'] = device.compute_module.overhead.matmul
        logs = dump_log(logs, 'a_mul_v', a_mul_v_latency, self.A_mul_V.stats, launch_overhead)

        print(f"a_mul_v latency: {a_mul_v_latency}, compute latency: {self.A_mul_V.stats.compute_latency}, io overhead: {self.A_mul_V.stats.io_latency}, kernel launch overhead: {device.compute_module.overhead.matmul}")

        print("simulating h_matmul0")
        h_matmul0_latency = (
            self.H_matmul0.compile_and_simulate(device, compile_mode=compile_mode, strategy=h_matmul0_mapping)
            + device.compute_module.overhead.matmul
        )
        # logs['h_matmul0'] = {}
        # logs['h_matmul0']['latency'] = h_matmul0_latency
        # logs['h_matmul0']['compute_latency'] = self.H_matmul0.stats.compute_latency
        # logs['h_matmul0']['io_latency'] = self.H_matmul0.stats.io_latency
        # logs['h_matmul0']['kernel_launch_overhead'] = device.compute_module.overhead.matmul


        logs = dump_log(logs, 'h_matmul0', h_matmul0_latency, self.H_matmul0.stats, launch_overhead)
        print(f"h_matmul0 latency: {h_matmul0_latency}, compute latency: {self.H_matmul0.stats.compute_latency}, io overhead: {self.H_matmul0.stats.io_latency}, kernel launch overhead: {device.compute_module.overhead.matmul}")

        print("simulating h1_matmul1")
        h1_matmul1_latency = (
            self.H_matmul1.compile_and_simulate(device, compile_mode=compile_mode, strategy=h1_matmul1_mapping)
            + device.compute_module.overhead.matmul
        )
        # logs['h1_matmul1'] = {}
        # logs['h1_matmul1']['latency'] = h1_matmul1_latency
        # logs['h1_matmul1']['compute_latency'] = self.H_matmul1.stats.compute_latency
        # logs['h1_matmul1']['io_latency'] = self.H_matmul1.stats.io_latency
        # logs['h1_matmul1']['kernel_launch_overhead'] = device.compute_module.overhead.matmul

        logs = dump_log(logs, 'h1_matmul1', h1_matmul1_latency, self.H_matmul1.stats, launch_overhead)
        print(f"h1_matmul1 latency: {h1_matmul1_latency}, compute latency: {self.H_matmul1.stats.compute_latency}, io overhead: {self.H_matmul1.stats.io_latency}, kernel launch overhead: {device.compute_module.overhead.matmul}")

        print("simulating h2_matmul2")
        h2_matmul2_latency = (
            self.H_matmul2.compile_and_simulate(device, compile_mode=compile_mode, strategy=h2_matmul2_mapping)
            + device.compute_module.overhead.matmul
        )
        # logs['h2_matmul2'] = {}
        # logs['h2_matmul2']['latency'] = h2_matmul2_latency
        # logs['h2_matmul2']['compute_latency'] = self.H_matmul2.stats.compute_latency
        # logs['h2_matmul2']['io_latency'] = self.H_matmul2.stats.io_latency
        # logs['h2_matmul2']['kernel_launch_overhead'] = device.compute_module.overhead.matmul

        logs = dump_log(logs, 'h2_matmul2', h2_matmul2_latency, self.H_matmul2.stats, launch_overhead)
        print(f"h2_matmul2 latency: {h2_matmul2_latency}, compute latency: {self.H_matmul2.stats.compute_latency}, io overhead: {self.H_matmul2.stats.io_latency}, kernel launch overhead: {device.compute_module.overhead.matmul}")
        print("finish matmul simulation")

        matmul_total_latency = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + h1_matmul1_latency
            + h2_matmul2_latency
        )

       

        print(f"matmul total overhead: {8*device.compute_module.overhead.matmul}")
        print(f"matmul total latency w/o overhead: {matmul_total_latency - 8*device.compute_module.overhead.matmul}")
        print(f"matmul total latency: {matmul_total_latency}")

        # normalization
        softmax_latency = (
            self.A_softmax.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.softmax
        )
        layernorm_latency = (
            self.layer_norm0.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.layernorm
        )

        normlization_total_latency = softmax_latency + layernorm_latency * 2

        # gelu
        gelu_latency = (
            self.H_gelu.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.gelu
        )

        # allreduce
        if self.device_count > 1:
            allreduce_latency = self.allreduce_mha.simulate(interconnect)
            allreduce_total_latency = allreduce_latency * 2
        else:
            allreduce_latency = 0
            allreduce_total_latency = 0

        #log latencies to a csv file
        with open('transformer_decode_profile_log.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Entry',
                'Q_K_V', 'Q_mul_k', 'A_mul_V', 
                'Wo_proj', 'W1_proj', 'W2_proj', 
                'Softmax', 'Layernorm', 
                'GeLU', 'Allreduce'
            ])
            # writer.writerow(['End2End Latency',
            #     qkv_latency, q_mul_k_latency, a_mul_v_latency, 
            #     h_matmul0_latency, h1_matmul1_latency, h2_matmul2_latency, 
            #     softmax_latency, layernorm_latency,
            #     gelu_latency, allreduce_latency
            # ])
            keys = ['entry','qkv', 'q_mul_k', 'a_mul_v', 'h_matmul0', 'h1_matmul1', 'h2_matmul2']

            logs['entry'] = {'latency':'latency','compute_latency': 'compute_latency', 'io_latency': 'io_latency', 
                             'kernel_launch_overhead': 'kernel_launch_overhead', 'tiling_utilization': 'tiling_utilization',
                             'simd_utilization': 'simd_utilization', 'capacity_utilization': 'capacity_utilization'}
            


            latency_row = [logs[key]['latency'] for key in keys]
            compute_latency_row = [logs[key]['compute_latency'] for key in keys]
            io_latency_row = [logs[key]['io_latency'] for key in keys]
            launch_overhead_row = [logs[key]['kernel_launch_overhead'] for key in keys]
            simd_utilization_row = [logs[key]['simd_utilization'] for key in keys]
            capacity_utilization_row = [logs[key]['capacity_utilization'] for key in keys]
            tiling_utilization_row = [logs[key]['tiling_utilization'] for key in keys]

            writer.writerow(latency_row)
            writer.writerow(compute_latency_row)
            writer.writerow(io_latency_row)
            writer.writerow(launch_overhead_row)
            writer.writerow(simd_utilization_row)
            writer.writerow(capacity_utilization_row)
            writer.writerow(tiling_utilization_row)

        # others

        # print
        # print("breakdown:")
        # print(
        #     f"{qkv_latency}\n{q_mul_k_latency}\n{a_mul_v_latency}\n{h_matmul0_latency}\n{h1_matmul1_latency}\n{h2_matmul2_latency}\n{softmax_latency}\n{layernorm_latency}\n{layernorm_latency}\n{gelu_latency}\n{allreduce_latency}\n{allreduce_latency}\n"
        # )
        # print("total:")
        # print(
        #     f"{matmul_total_latency}\n{normlization_total_latency}\n{gelu_latency}\n{allreduce_total_latency}\n"
        # )
        self.latency = (
            matmul_total_latency
            + normlization_total_latency
            + gelu_latency
            + allreduce_total_latency
        )
        self.simulate_log = f"{qkv_latency}, {q_mul_k_latency}, {a_mul_v_latency}, {h_matmul0_latency}, {h1_matmul1_latency}, {h2_matmul2_latency}, {softmax_latency}, {layernorm_latency}, {layernorm_latency}, {gelu_latency}, {allreduce_latency}, {allreduce_latency}"
        return self.latency
    
    def compile_and_simulate_gpu(self, system: System, compile_mode: str):
        pcb = system.device
        core_count_per_block = self.core_count_per_block
        interconnect = system.interconnect

        
        # matmul
        print("simulating qkv")
        qkv_latency = 3 * (
            self.Q_proj.compile_and_simulate(pcb, compile_mode)
            + pcb.compute_module.overhead.matmul
        )
        print(f"qkv latency: {qkv_latency}")
        print("simulating q_mul_k")
        q_mul_k_latency = (
            self.Q_mul_K.compile_and_simulate(pcb, compile_mode)
            + pcb.compute_module.overhead.matmul
        )
        print(f"q_mul_k latency: {q_mul_k_latency}")
        print("simulating a_mul_v")
        a_mul_v_latency = (
            self.A_mul_V.compile_and_simulate(pcb, compile_mode)
            + pcb.compute_module.overhead.matmul
        )
        print(f"a_mul_v latency: {a_mul_v_latency}")
        print("simulating h_matmul0")
        h_matmul0_latency = (
            self.H_matmul0.compile_and_simulate(pcb, compile_mode)
            + pcb.compute_module.overhead.matmul
        )
        print(f"h_matmul0 latency: {h_matmul0_latency}")
        print("simulating h1_matmul1")
        h1_matmul1_latency = (
            self.H_matmul1.compile_and_simulate(pcb, compile_mode)
            + pcb.compute_module.overhead.matmul
        )
        print(f"h1_matmul1 latency: {h1_matmul1_latency}")
        print("simulating h2_matmul2")
        h2_matmul2_latency = (
            self.H_matmul2.compile_and_simulate(pcb, compile_mode)
            + pcb.compute_module.overhead.matmul
        )
        print(f"h2_matmul2 latency: {h2_matmul2_latency}")

        matmul_total_latency = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + h1_matmul1_latency
            + h2_matmul2_latency
        )
        print(f"matmul total overhead: {8*pcb.compute_module.overhead.matmul}")
        print(f"matmul total latency w/o overhead: {matmul_total_latency - 8*pcb.compute_module.overhead.matmul}")
        print(f"matmul total latency: {matmul_total_latency}")

        # normalization
        softmax_latency = (
            self.A_softmax.compile_and_simulate(pcb, compile_mode)
            + pcb.compute_module.overhead.softmax
        )
        layernorm_latency = (
            self.layer_norm0.compile_and_simulate(pcb, compile_mode)
            + pcb.compute_module.overhead.layernorm
        )

        normlization_total_latency = softmax_latency + layernorm_latency * 2

        # gelu
        gelu_latency = (
            self.H_gelu.compile_and_simulate(pcb, compile_mode)
            + pcb.compute_module.overhead.gelu
        )

        # allreduce
        if self.device_count > 1:
            allreduce_latency = self.allreduce_mha.simulate(interconnect)
            allreduce_total_latency = allreduce_latency * 2
        else:
            allreduce_latency = 0
            allreduce_total_latency = 0

        # others

        # print
        # print("breakdown:")
        # print(
        #     f"{qkv_latency}\n{q_mul_k_latency}\n{a_mul_v_latency}\n{h_matmul0_latency}\n{h1_matmul1_latency}\n{h2_matmul2_latency}\n{softmax_latency}\n{layernorm_latency}\n{layernorm_latency}\n{gelu_latency}\n{allreduce_latency}\n{allreduce_latency}\n"
        # )
        # print("total:")
        # print(
        #     f"{matmul_total_latency}\n{normlization_total_latency}\n{gelu_latency}\n{allreduce_total_latency}\n"
        # )
        self.latency = (
            matmul_total_latency
            + normlization_total_latency
            + gelu_latency
            + allreduce_total_latency
        )
        self.simluate_log = f"{qkv_latency}, {q_mul_k_latency}, {a_mul_v_latency}, {h_matmul0_latency}, {h1_matmul1_latency}, {h2_matmul2_latency}, {softmax_latency}, {layernorm_latency}, {layernorm_latency}, {gelu_latency}, {allreduce_latency}, {allreduce_latency}"
        return self.latency
    
    def compile_and_simulate(self, system: System, compile_mode: str):
        device = system.device
        if device.type == "simdram":
            return self.compile_and_simulate_simdram(system, compile_mode)
        else:
            return self.compile_and_simulate_gpu(system, compile_mode)


    def run_on_gpu(self):
        # matmul
        qkv_latency = (
            self.Q_proj.run_on_gpu()  # - self.Q_proj.gpu_kernel_launch_overhead()
        ) * 3
        q_mul_k_latency = (
            self.Q_mul_K.run_on_gpu()  # - self.Q_mul_K.gpu_kernel_launch_overhead()
        )
        a_mul_v_latency = (
            self.A_mul_V.run_on_gpu()  # - self.A_mul_V.gpu_kernel_launch_overhead()
        )
        h_matmul0_latency = (
            self.H_matmul0.run_on_gpu()  # - self.H_matmul0.gpu_kernel_launch_overhead()
        )
        h1_matmul1_latency = (
            self.H_matmul1.run_on_gpu()  # - self.H_matmul1.gpu_kernel_launch_overhead()
        )
        h2_matmul2_latency = (
            self.H_matmul2.run_on_gpu()  # - self.H_matmul2.gpu_kernel_launch_overhead()
        )

        matmul_total_latency = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + h1_matmul1_latency
            + h2_matmul2_latency
        )

        # normalization
        softmax_latency = (
            self.A_softmax.run_on_gpu()  # - self.A_softmax.gpu_kernel_launch_overhead()
        )
        layernorm_latency = (
            self.layer_norm0.run_on_gpu()
            - self.layer_norm0.gpu_kernel_launch_overhead()
        )

        normlization_total_latency = softmax_latency + layernorm_latency * 2

        # gelu
        gelu_latency = (
            self.H_gelu.run_on_gpu()  # - self.H_gelu.gpu_kernel_launch_overhead()
        )
        # gelu_latency = max(gelu_latency, 1e-7)

        # allreduce
        allreduce_total_latency = 0

        # others

        # print
        print("breakdown:")
        print(
            f"{qkv_latency}\n{q_mul_k_latency}\n{a_mul_v_latency}\n{h_matmul0_latency}\n{h1_matmul1_latency}\n{h2_matmul2_latency}\n{softmax_latency}\n{layernorm_latency}\n{layernorm_latency}\n{gelu_latency}\n"
        )
        print("total:")
        print(
            f"{matmul_total_latency}\n{normlization_total_latency}\n{gelu_latency}\n{allreduce_total_latency}\n"
        )
        self.latency_on_gpu = (
            matmul_total_latency
            + normlization_total_latency
            + gelu_latency
            + allreduce_total_latency
        )
        return self.latency_on_gpu

class LLMInitComputationTP:
    def __init__(
        self,
        d_model,
        n_heads,
        n_layers,
        device_count,
    ) -> None:
        pass
