from design_space_exploration.dse import template_to_system, template_to_system_pimsab, read_architecture_template
from software_model.transformer import (
    TransformerBlockInitComputationTP,
    TransformerBlockAutoRegressionTP,
)
from software_model.utils import data_type_dict, Tensor
from software_model.transformer_hyper import Transformer_hyper
import argparse
parser = argparse.ArgumentParser(description="print system info")
parser.add_argument("--config", type=str, help="Path to the config file")
args = parser.parse_args()

specs = read_architecture_template(args.config)
system = template_to_system(specs)
simdram = system.device
print(simdram.info())