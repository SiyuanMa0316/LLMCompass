from math import ceil
from software_model.utils import DataType, data_type_dict


class Tile:
    def __init__(
        self,
        arr_count,
        arr_cols,
        arr_rows,
        
    ):
        self.arr_count = arr_count
        self.arr_cols = arr_cols
        self.arr_rows = arr_rows
        


tile_dict = {
    "Tile_1024x128x128": Tile(1024, 128, 128),
    "Tile_256x256x256": Tile(256, 256, 256),
    "Tile_512x256x128": Tile(512, 256, 128),
}




class Overhead:
    def __init__(self, matmul, softmax, layernorm, gelu):
        self.matmul = matmul
        self.softmax = softmax
        self.layernorm = layernorm
        self.gelu = gelu


overhead_dict = {
    "A100": Overhead(2.1e-5, 1.2e-5, 4.5e-5, 4.5e-5),
    "TPUv3": Overhead(11e-5, 30e-5, 14e-5, 10e-5),
    "MI210": Overhead(3.4e-5, 2.2e-5, 2.8e-5, 2.1e-5),
    "PIMSAB": Overhead(2.1e-5, 1.2e-5, 4.5e-5, 4.5e-5),
}

class NoC:
    def __init__(self, 
        bit_width,
        freq
    ):
        self.bit_width = bit_width #bits
        self.freq = freq #Hz
        self.bandwidth = bit_width * freq / 8 #bytes per second

noc_dict = {
    "NoC_1024b": NoC(1024, 1.5e9),
}
class ComputeModulePIMSAB:
    def __init__(
        self,
        tile: Tile,
        tile_count,
        noc: NoC,
        clock_freq,
        overhead: Overhead = overhead_dict["PIMSAB"],
    ):
        self.tile = tile
        self.tile_count = tile_count
        self.clock_freq = clock_freq
        self.noc = noc
        self.overhead = overhead


compute_module_pimsab_dict = {
    "PIMSAB_12x10_256x256x256": ComputeModulePIMSAB(
        tile_dict["Tile_256x256x256"],
        120,
        noc_dict["NoC_1024b"],
        1.5e9,
        overhead_dict["A100"],
    ),
    "PIMSAB_12x10_1024x128x128": ComputeModulePIMSAB(
        tile_dict["Tile_1024x128x128"],
        120,
        noc_dict["NoC_1024b"],
        1.5e9,
        overhead_dict["A100"],
    ),
    "PIMSAB_12x10_512x256x128": ComputeModulePIMSAB(
        tile_dict["Tile_512x256x128"],
        120,
        noc_dict["NoC_1024b"],
        1.5e9,
        overhead_dict["A100"],
    ),
}
