from typing import Dict
def get_dup_in_arr(arr_mapping: Dict[str, str], arr_tile_M: int, arr_tile_K: int, arr_tile_N: int):
    if arr_mapping['C'] == 'M':
        MK_dup = 1
        KN_dup = arr_tile_M
        MN_dup = 1
    elif arr_mapping['C'] == 'N':
        MK_dup = arr_tile_N
        KN_dup = 1
        MN_dup = 1
    elif arr_mapping['C'] == 'K':
        MK_dup = 1
        KN_dup = 1
        MN_dup = arr_tile_K
    elif arr_mapping['C'] == 'MN' or arr_mapping['C'] == 'NM':
        MK_dup = arr_tile_N
        KN_dup = arr_tile_M
        MN_dup = 1
    elif arr_mapping['C'] == 'MK' or arr_mapping['C'] == 'KM':
        MK_dup = 1
        KN_dup = arr_tile_M
        MN_dup = arr_tile_K
    elif arr_mapping['C'] == 'NK' or arr_mapping['C'] == 'KN':
        MK_dup = arr_tile_N
        KN_dup = 1
        MN_dup = arr_tile_K
    else:
        raise ValueError(f"Invalid arr_mapping: {arr_mapping}")
    return MK_dup, KN_dup, MN_dup