from hardware_model.compute_module import ComputeModule, compute_module_dict
from hardware_model.compute_module_pimsab import ComputeModulePIMSAB, compute_module_pimsab_dict
from hardware_model.compute_module_simdram import ComputeModuleSIMDRAM, compute_module_simdram_dict
from hardware_model.io_module import IOModule, IO_module_dict
from hardware_model.memory_module import MemoryModule, memory_module_dict
from typing import Union

class Device:
    def __init__(
        self,
        type: str,
        compute_module: Union[ComputeModule, ComputeModulePIMSAB, ComputeModuleSIMDRAM],
        io_module: IOModule,
        memory_module: MemoryModule,
    ) -> None:
        self.type = type
        self.compute_module = compute_module
        self.io_module = io_module
        self.memory_module = memory_module


device_dict = {
    "A100_80GB_fp16": Device(
        "systolic",
        compute_module_dict["A100_fp16"],
        IO_module_dict["A100"],
        memory_module_dict["A100_80GB"],
    ),
    "TPUv3": Device(
        "systolic",
        compute_module_dict["TPUv3_bf16"],
        IO_module_dict["TPUv3"],
        memory_module_dict["TPUv3"],
    ),
    "MI210": Device(
        "systolic",
        compute_module_dict["MI210_fp16"],
        IO_module_dict["MI210"],
        memory_module_dict["MI210"],
    ),
    "TPUv3_new": Device(
        "systolic",
        compute_module_dict["TPUv3_new"],
        IO_module_dict["TPUv3"],
        memory_module_dict["TPUv3"],
    ),
    "PIMSAB_12x10_256x256x256": Device(
        "pimsab",
        compute_module_pimsab_dict["PIMSAB_12x10_256x256x256"],
        IO_module_dict["PIMSAB"],
        memory_module_dict["PIMSAB"],
    ),
    "PIMSAB_12x10_1024x128x128": Device(
        "pimsab",
        compute_module_pimsab_dict["PIMSAB_12x10_1024x128x128"],
        IO_module_dict["PIMSAB"],
        memory_module_dict["PIMSAB"],
    ),
    "PIMSAB_12x10_512x256x128": Device(
        "pimsab",
        compute_module_pimsab_dict["PIMSAB_12x10_512x256x128"],
        IO_module_dict["PIMSAB"],
        memory_module_dict["PIMSAB"],
    ),
    "SIMDRAM_STD": Device(
        "simdram",
        compute_module_simdram_dict["simdram_standard"],
        IO_module_dict["SIMDRAM"],
        memory_module_dict["SIMDRAM"],
    ),
}
