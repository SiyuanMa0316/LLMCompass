from design_space_exploration.dse import template_to_system, template_to_system_pimsab, read_architecture_template
from software_model.transformer_Llama import (
    TransformerBlockInitComputationTP,
    TransformerBlockAutoRegressionTP,
)
from software_model.transformer_hyper import Transformer_hyper
from software_model.utils import data_type_dict, Tensor

specs = read_architecture_template("configs/SIMDRAM_96x_arr512.json")
system = template_to_system(specs)
simdram = system.device
print(simdram.info())

llm_hyper = Transformer_hyper()
llm_hyper.read_from_json("LLM_hyper/llama-3.1-70b.json")
layers = llm_hyper.num_layers
bs=1
seq_len=1024
precision = "int8"
model_prefill = TransformerBlockInitComputationTP(
        d_model=llm_hyper.d_model,
        n_heads=llm_hyper.num_heads,
        n_kv_heads=llm_hyper.num_kv_heads,
        ffn_dim=llm_hyper.ffn_dim,
        device_count=1,
        data_type=data_type_dict["int8"],
    )
_ = model_prefill(
	Tensor([bs, seq_len, llm_hyper.d_model], data_type_dict["int8"])
)
prefill_latency_simulated = model_prefill.compile_and_simulate(system, compile_mode = "specific")

E2E_prefill_latency = prefill_latency_simulated * layers
print(f"Llama-3 {layers} layers prefill latency: {E2E_prefill_latency}")

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