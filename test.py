from design_space_exploration.dse import template_to_system, read_architecture_template
from software_model.utils import Tensor, DataType, data_type_dict
from software_model.transformer import TransformerBlockAutoRegressionTP, TransformerBlockInitComputationTP

specs = read_architecture_template("configs/GA100.json")
sys = template_to_system(specs)

model_auto_regression = TransformerBlockAutoRegressionTP(
    d_model = 12288,
    n_heads = 96,
    device_count = 1,
    data_type = data_type_dict['fp16'],
)

bs = 16
seq_len = 1024

_ = model_auto_regression(
    Tensor([bs, 1, 12288], data_type_dict['fp16']),
    seq_len,
)

auto_regression_latency_simulated = model_auto_regression.compile_and_simulate(
    sys, "heuristic-GPU"
)

