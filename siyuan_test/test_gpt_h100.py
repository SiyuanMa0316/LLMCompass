from design_space_exploration.dse import template_to_system, template_to_system_pimsab, read_architecture_template
from software_model.transformer import (
    TransformerBlockInitComputationTP,
    TransformerBlockAutoRegressionTP,
)
from software_model.utils import data_type_dict, Tensor
from software_model.transformer_hyper import Transformer_hyper
import argparse
parser = argparse.ArgumentParser(description="Test GPT-3 on H100")
# parser.add_argument("--config", type=str, help="Path to the config file")
parser.add_argument("--model", type=str, help="Path to model hyperparameter file")
args = parser.parse_args()

llm_hyper = Transformer_hyper()
llm_hyper.read_from_json(args.model)
layers = llm_hyper.num_layers
bs=1
seq_len=1024
precision = "int8"
specs = read_architecture_template(f"configs/H100x1_{precision}.json")
system = template_to_system(specs)
print(f"memory bandwidth: {system.device.io_module.bandwidth}B/s")
print(f"flops: {system.device.compute_module.total_systolic_array_flops/1e12}TFLOPS")
model_prefill = TransformerBlockInitComputationTP(
        d_model=llm_hyper.d_model,
        n_heads=llm_hyper.num_heads,
        device_count=1,
        data_type=data_type_dict[precision],
    )
_ = model_prefill(
	Tensor([bs, seq_len, llm_hyper.d_model], data_type_dict[precision])
)
prefill_latency_simulated = model_prefill.compile_and_simulate(
	system, "heuristic-GPU"
)

E2E_prefill_latency = prefill_latency_simulated * layers
print(f"GPT-3 {layers} layers prefill latency: {E2E_prefill_latency}")

output_len = 2048
model_decode = TransformerBlockAutoRegressionTP(
        d_model=llm_hyper.d_model,
        n_heads=llm_hyper.num_heads,
        device_count=1,
        data_type=data_type_dict[precision],
    )
_ = model_decode(
    Tensor([bs, 1, llm_hyper.d_model], data_type_dict[precision]), seq_len
)
decode_latency_simulated = model_decode.compile_and_simulate(
    system, "heuristic-GPU"
) * layers
print(f"GPT-3 decode latency per token: {decode_latency_simulated}")

latency_to_first_token = E2E_prefill_latency + decode_latency_simulated
decode_tokps = 1 / decode_latency_simulated
total_latency = E2E_prefill_latency +  decode_latency_simulated * output_len
print(f"Summary: ")
print(f"GPT-3 prefill latency ({seq_len} tokens, 1 layer): {prefill_latency_simulated}")
print(f"GPT-3 prefill latency ({seq_len} tokens, {layers} layers): {E2E_prefill_latency}")
print(f"GPT-3 decode latency per token: {decode_latency_simulated}")
print(f"GPT-3 latency to first token: {latency_to_first_token}")
print(f"GPT-3 tok/s: {decode_tokps}")
print(f"GPT-3 total latency ({output_len} output tokens): {total_latency}")