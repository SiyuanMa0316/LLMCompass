from design_space_exploration.dse import template_to_system, template_to_system_pimsab, read_architecture_template
from software_model.transformer_Llama import (
    TransformerBlockInitComputationTP,
    TransformerBlockAutoRegressionTP,
)
from software_model.transformer_hyper import Transformer_hyper
from software_model.utils import data_type_dict, Tensor

specs = read_architecture_template("configs/SIMDRAM_96x_bitserial.json")
system = template_to_system(specs)
simdram = system.device
print(simdram.info())

# llm_hyper = Transformer_hyper()
# llm_hyper.read_from_json("LLM_hyper/llama-3.1-70b.json")
# layers = llm_hyper.num_layers
# bs=1
# seq_len=1024
# precision = "int8"
# model_prefill = TransformerBlockInitComputationTP(
#         d_model=llm_hyper.d_model,
#         n_heads=llm_hyper.num_heads,
#         n_kv_heads=llm_hyper.num_kv_heads,
#         ffn_dim=llm_hyper.ffn_dim,
#         device_count=1,
#         data_type=data_type_dict["int8"],
#     )
# _ = model_prefill(
# 	Tensor([bs, seq_len, llm_hyper.d_model], data_type_dict["int8"])
# )
# prefill_latency_simulated = model_prefill.compile_and_simulate(system, compile_mode = "specific")

# E2E_prefill_latency = prefill_latency_simulated * layers
# print(f"Llama-3 {layers} layers prefill latency: {E2E_prefill_latency}")

