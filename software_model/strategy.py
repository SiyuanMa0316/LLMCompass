from hardware_model.device import Device
class Strategy:
    def __init__(self, tile_mapping: dict, arr_mapping: dict, 
                 loop_order: str='mkn', PE_enable = False, broadcast: str = "AB", arr_multicast=False, col_popcount=True, weight_resident = False, input_resident=False, output_resident=False) -> None:
        self.tile_mapping = tile_mapping
        self.arr_mapping = arr_mapping
        self.loop_order = loop_order
        self.with_PE = PE_enable
        self.broadcast = broadcast
        self.arr_multicast = arr_multicast
        self.col_popcount = col_popcount
        self.weight_resident = weight_resident
        self.input_resident = input_resident
        self.output_resident = output_resident
        
    def __str__(self) -> str:
        tile_mapping_str = "".join([f"  {key}: {value}" for key, value in self.tile_mapping.items()])
        arr_mapping_str = "".join([f"  {key}: {value}" for key, value in self.arr_mapping.items()])
        return (f"Strategy(\n"
                f"  Loop Order: {self.loop_order}"
                f"  PE Enabled: {self.with_PE}"
                f"  Broadcast: {self.broadcast}\n"
                f"  Array Multicast: {self.arr_multicast}\n"
                f"  Column Popcount: {self.col_popcount}\n"
                f"  Tile Mapping:{tile_mapping_str}"
                f"  Array Mapping:{arr_mapping_str}\n"
                f"  Weight Resident: {self.weight_resident}\n"
                f"  Input Resident: {self.input_resident}\n"
                f"  Output Resident: {self.output_resident}\n)")
    
    def to_json(self, file_path=None):
        json = {
            "tile_mapping": self.tile_mapping,
            "arr_mapping": self.arr_mapping,
            "loop_order": self.loop_order,
            "with_PE": self.with_PE,
            "broadcast": self.broadcast,
            "arr_multicast": self.arr_multicast,
            "col_popcount": self.col_popcount,
            "weight_resident": self.weight_resident,
            "input_resident": self.input_resident,
            "output_resident": self.output_resident
        }
        if file_path:
            import json as js
            with open(file_path, 'w') as f:
                js.dump(json, f, indent=4)
        return json

    @staticmethod
    def tile_mapping_extraction(device:Device, s):
        result = {'M': None, 'N': None, 'K': None}
        # Ensure all characters are unique and meet the required conditions
        chars = set(s)
        # required = {'A', 'B', 'D', 'R', 'C'} # array, bank, device, rank, channel
        required = set(device.compute_module.parallelisms.keys())
        assert required.issubset(chars), "A, B, D, R, C must be present"
        assert 2 <= len(chars & {'M', 'N', 'K'}) <= 3, "At least two of M, N, K must be present"
        assert len(s) == len(chars), f"{s}: All characters must be unique"

        for target in ['M', 'N', 'K']:
            if target not in s:
                continue  # Keep the default value of None
            idx = s.index(target)
            matching = []
            # Search forward until a non-A/B/D character or another M/N/K is encountered
            for i in range(idx + 1, len(s)):
                char = s[i]
                if char in required:
                    matching.append(char)
                elif char in {'M', 'N', 'K'}:
                    break  # Stop when encountering another M/N/K
                else:
                    assert False, "Invalid character"  # Should never occur according to input constraints
            result[target] = ''.join(matching) if matching else ''
        return result
    
    @staticmethod
    def arr_mapping_extraction(s):
        result = {'R': '', 'C': ''}
        # Ensure all characters are unique and meet the required conditions
        chars = set(s)
        required = {'M', 'K', 'N'}
        assert required.issubset(chars), "M, K, N must be present"
        assert 1 <= len(chars & {'R', 'C'}) <= 2, "At least two of M, N, K must be present"
        assert len(s) == len(chars), f"{s}: All characters must be unique"

        for target in ['R', 'C']:
            if target not in s:
                continue  # Keep the default value of None
            idx = s.index(target)
            matching = []
            # Search forward until a non-A/B/D character or another M/N/K is encountered
            for i in range(idx + 1, len(s)):
                char = s[i]
                if char in {'M', 'K', 'N'}:
                    matching.append(char)
                elif char in {'R', 'C'}:
                    break  # Stop when encountering another M/N/K
                else:
                    assert False, "Invalid character"  # Should never occur according to input constraints
            result[target] = ''.join(matching) if matching else ''
        return result
    
    def get_mapping_from_json(json_file_path: str):
        import json as json
        with open(json_file_path, 'r') as f:
            json_data = json.load(f)
            tile_mapping = json_data.get("tile_mapping")
            arr_mapping = json_data.get("arr_mapping")
            loop_order = json_data.get("loop_order", 'mkn')
            PE_enable = json_data.get("with_PE", False)
            broadcast = json_data.get("broadcast", "AB")
            arr_multicast = json_data.get("arr_multicast", False)
            col_popcount = json_data.get("col_popcount", True)
            weight_resident = json_data.get("weight_resident", False)
            input_resident = json_data.get("input_resident", False)
            output_resident = json_data.get("output_resident", False)
            return Strategy(tile_mapping, arr_mapping, loop_order, PE_enable, broadcast, arr_multicast, col_popcount, weight_resident, input_resident, output_resident)

    

if __name__ == '__main__':
    # Demo testcases
    s = "MKNABD"
    res = Strategy.tile_mapping_extraction(s)
    print(res)
    s= "RMNCK"
    res = Strategy.arr_mapping_extraction(s)
    print(res)
    # assert res == expect, "Extracted result does not match"