from design_space_exploration.dse import template_to_system, template_to_system_pimsab, read_architecture_template
from software_model.transformer_Llama import (
    TransformerBlockInitComputationTP,
    TransformerBlockAutoRegressionTP,
)
from software_model.transformer_hyper import Transformer_hyper
from software_model.utils import data_type_dict, Tensor
import argparse
parser = argparse.ArgumentParser(description="Test Llama on DRAM PIM")
parser.add_argument("--config", type=str, help="Path to the config file")
args = parser.parse_args()

specs = read_architecture_template(args.config)
system = template_to_system(specs)
simdram = system.device
simdram.compute_module.channel_count = 1
simdram.compute_module.rank_count = 1
print('SimDRAM configuration for each pipeline stage:')
print(simdram.info())

llm_hyper = Transformer_hyper()
llm_hyper.read_from_json("LLM_hyper/llama-3.1-70b.json")
layers = llm_hyper.num_layers
bs=1
seq_len=1024
precision = "int8"
# model_prefill = TransformerBlockInitComputationTP(
#         d_model=12288,
#         n_heads=96,
#         device_count=1,
#         data_type=data_type_dict["int8"],
#     )
# _ = model_prefill(
# 	Tensor([bs, seq_len, 12288], data_type_dict["int8"])
# )
# prefill_latency_simulated = model_prefill.compile_and_simulate(system, compile_mode = "specific")

# E2E_prefill_latency = prefill_latency_simulated * layers
# print(f"GPT-3 {layers} layers prefill latency: {E2E_prefill_latency}")

output_len = 2048
model_decode = TransformerBlockAutoRegressionTP(
        d_model=llm_hyper.d_model,
        n_kv_heads=llm_hyper.num_kv_heads,
        n_heads=llm_hyper.num_heads,
        ffn_dim=llm_hyper.ffn_dim,
        device_count=1,
        data_type=data_type_dict["int8"],
    )
_ = model_decode(
    Tensor([bs, 1, llm_hyper.d_model], data_type_dict["int8"]), seq_len
)
decode_latency_simulated = model_decode.compile_and_simulate(system, "specific")
print(f"GPT-3 decode latency per token: {decode_latency_simulated}")
decode_total_latency = decode_latency_simulated * layers + decode_latency_simulated * (output_len-1)
print(f"GPT-3 decode total latency for {output_len} tokens: {decode_total_latency}")

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