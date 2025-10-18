import csv

workloads = ['2mm', '3mm', 'GEMM_1024x12288x12288', 'GEMM_2048x24576x24576', 'GEMV_1x12288x12288', 'GEMV_1x24576x24576', 'gpt3-175B_prefill', 'gpt3-175B_decode', 'gpt3-6.7B_prefill', 'gpt3-6.7B_decode', 'Llama-3.1-70B_prefill', 'Llama-3.1-70B_decode', 'Llama-3.1-8B_prefill', 'Llama-3.1-8B_decode']

#ms
proteus_gemm_latencies={
    "2048x12288x12288": 191134.42,
    "2048x128x2048": 1990.64,
    "2048x2048x128": 31785.76,
    "2048x12288x49152": 191134.43,
    "2048x49152x12288": 764537.70,
    "1x12288x12288": 88.74,
    "1x128x2049": 0.92,
    "1x2049x128": 14.79,
    "1x12288x49152": 93.33,
    "1x49152x12288": 354.95,
    "1024x4096x4096": 0,
    "1024x4096x1024": 31855.39,
    "1024x128x1024": 995.14,
    "1024x1024x128": 7946.36,
    "1024x4096x14336": 31855.73,
    "1024x14336x4096": 111495.04,
    "1x4096x4096": 29.58,
    "1x4096x1024": 29.58,
    "1x128x1025": 0.92,
    "1x1025x128": 7.40,
    "1x4096x4096": 29.58,
    "1x4096x14336": 29.58,
    "1024x14336x4096": 111495.04,
    "1x14336x4096": 103.53,
    "1024x8192x8192": 63711.13,
    "1024x8192x1024": 63711.13,
    "1024x8192x28672": 63711.13,
    "1024x28672x8192": 222990.14,
    "1x8192x8192": 59.16,
    "1x8192x1024": 59.16,
    "1x8192x28672": 59.16,
    "1x28672x8192": 207.05
}

from software_model.utils import data_type_dict, Tensor
from software_model.transformer_hyper import Transformer_hyper
import argparse
parser = argparse.ArgumentParser(description="Test LLM on Proteus")
parser.add_argument("--config", type=str, help="Path to the config file")
parser.add_argument("--model", type=str, help="Path to model hyperparameter file")
parser.add_argument("--prefill", action='store_true', help="Run prefill phase")
parser.add_argument("--decode", action='store_true', help="Run decode phase")
args = parser.parse_args()


print("Simulate Running on Proteus")

# Load model hyperparameters
llm_hyper = Transformer_hyper()
llm_hyper.read_from_json(args.model)
if "gpt" in llm_hyper.name:
    from software_model.transformer import (
        TransformerBlockInitComputationTP,
        TransformerBlockAutoRegressionTP,
    )
elif "Llama" in llm_hyper.name:
    from software_model.transformer_Llama import (
        TransformerBlockInitComputationTP,
        TransformerBlockAutoRegressionTP,
    )
layers = llm_hyper.num_layers
bs=1
seq_len=1024
precision = args.precision

# Prefill phase create model and simulate
if args.prefill:
    print(f"LLM model: {llm_hyper.name}")
    if "gpt" in llm_hyper.name:
        model_prefill = TransformerBlockInitComputationTP(
            model_name=llm_hyper.name,
            d_model=llm_hyper.d_model,
            n_heads=llm_hyper.num_heads,
            device_count=1,
            data_type=data_type_dict[precision],
        )
    elif "Llama" in llm_hyper.name:
        model_prefill = TransformerBlockInitComputationTP(
            model_name=llm_hyper.name,
            d_model=llm_hyper.d_model,
            n_kv_heads=llm_hyper.num_kv_heads,
            n_heads=llm_hyper.num_heads,
            ffn_dim=llm_hyper.ffn_dim,
            device_count=1,
            data_type=data_type_dict[precision],
        )
    _ = model_prefill(
        Tensor([bs, seq_len, llm_hyper.d_model], data_type_dict[precision])
    )

    prefill_latency_simulated = model_prefill.compile_and_simulate(system, "heuristic-GPU")


    E2E_prefill_latency = prefill_latency_simulated * layers
    print(f"{llm_hyper.name} {layers} layers prefill latency: {E2E_prefill_latency}")
    print(f"simulated latency: {llm_hyper.name}_prefill {E2E_prefill_latency}")

# Decode phase create model and simulate
if args.decode:
    if args.pipelined:
        simdram.compute_module.channel_count = 1
        simdram.compute_module.rank_count = 1
    output_len = 2048
    if "gpt" in llm_hyper.name:
        model_decode = TransformerBlockAutoRegressionTP(
                model_name=llm_hyper.name,
                d_model=llm_hyper.d_model,
                n_heads=llm_hyper.num_heads,
                device_count=1,
                data_type=data_type_dict[precision],
            )
    elif "Llama" in llm_hyper.name:
        model_decode = TransformerBlockAutoRegressionTP(
                model_name=llm_hyper.name,
                d_model=llm_hyper.d_model,
                n_kv_heads=llm_hyper.num_kv_heads,
                n_heads=llm_hyper.num_heads,
                ffn_dim=llm_hyper.ffn_dim,
                device_count=1,
                data_type=data_type_dict[precision],
            )
    _ = model_decode(
        Tensor([bs, 1, llm_hyper.d_model], data_type_dict[precision]), seq_len
    )
    if isGPU:
        decode_latency_simulated = model_decode.compile_and_simulate(system, "heuristic-GPU")
    else:
        decode_latency_simulated = model_decode.compile_and_simulate(system, "specific")
    print(f"{llm_hyper.name} decode latency per token: {decode_latency_simulated}")
    if args.pipelined:
        decode_total_latency = decode_latency_simulated * layers + decode_latency_simulated * (output_len-1)
    else:
        decode_total_latency = decode_latency_simulated * layers * output_len
    print(f"{llm_hyper.name} decode total latency for {output_len} tokens: {decode_total_latency}")
    print(f"simulated latency: {llm_hyper.name}_decode {decode_total_latency}")

# output_len = 2048
# model_decode = TransformerBlockAutoRegressionTP(
#         d_model=12288,
#         n_heads=96,
#         device_count=1,
#         data_type=data_type_dict["int4"],
#     )
# _ = model_decode(
#     Tensor([bs, 1, 12288], data_type_dict["int4"]), seq_len
# )
# decode_latency_simulated = model_decode.compile_and_simulate(
#     system, "heuristic-PIMSAB-sim-v2"
# )
# print(f"GPT-3 decode latency per token (1 layer mapped to 1 tile): {decode_latency_simulated}")

# latency_to_first_token = E2E_prefill_latency + decode_latency_simulated * layers
# decode_tokps = 1 / decode_latency_simulated
# total_latency = E2E_prefill_latency +  decode_latency_simulated * layers + decode_latency_simulated * output_len
# # print(f"{total_latency} = {E2E_prefill_latency} + {decode_latency_simulated} * {layers} + {decode_latency_simulated} * {output_len}")
# print(f"Summary: ")
# print(f"GPT-3 prefill latency ({seq_len} tokens, 1 layer): {prefill_latency_simulated}")
# print(f"GPT-3 prefill latency ({seq_len} tokens, {layers} layers): {E2E_prefill_latency}")
# print(f"GPT-3 decode latency per token: {decode_latency_simulated}")
# print(f"GPT-3 latency to first token: {latency_to_first_token}")
# print(f"GPT-3 tok/s: {decode_tokps}")
# print(f"GPT-3 total latency ({output_len} output tokens): {total_latency}")