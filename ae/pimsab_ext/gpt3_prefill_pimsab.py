from design_space_exploration.dse import template_to_system, read_architecture_template
from software_model.transformer import (
    TransformerBlockInitComputationTP,
    TransformerBlockAutoRegressionTP,
)
from software_model.utils import data_type_dict, Tensor

specs = read_architecture_template("configs/PIMSAB.json")
system = template_to_system(specs)

bs=1
seq_len=1024
model_prefill = TransformerBlockInitComputationTP(
        d_model=12288,
        n_heads=96,
        device_count=1,
        data_type=data_type_dict["int8"],
    )
_ = model_prefill(
	Tensor([bs, seq_len, 12288], data_type_dict["int8"])
)
prefill_latency_simulated = model_prefill.compile_and_simulate(
	system, "heuristic-PIMSAB-sim"
)
print(f"GPT-3 prefill latency: {prefill_latency_simulated}")

# output_len = 2048
# model_decode = TransformerBlockAutoRegressionTP(
#         d_model=12288,
#         n_heads=96,
#         device_count=1,
#         data_type=data_type_dict["int8"],
#     )
# _ = model_decode(
#     Tensor([bs, 1, 12288], data_type_dict["int8"]), seq_len
# )
# decode_latency_simulated = model_decode.compile_and_simulate(
#     system, "heuristic-PIMSAB"
# )
# print(f"GPT-3 decode latency per token: {decode_latency_simulated}")

# print(f"summary: GPT-3 prefill latency ({seq_len} tokens): {prefill_latency_simulated}, GPT-3 decode latency per token: {decode_latency_simulated}, total latency ({output_len} output tokens): {prefill_latency_simulated + decode_latency_simulated * output_len}")