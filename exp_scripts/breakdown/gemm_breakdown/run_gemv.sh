#!/bin/bash
cd ../../..
python -m experiments.test_gemm_ddr5_multiprocess --config configs/x16_base_config.json --M 1 --K 49152 --N 12288 --precision int8 --use_strategy exp_scripts/breakdown/gemm_breakdown/gemv_strategy.json
cd -
#configs/x16\{\'C\'\:\ 10\,\ \'R\'\:\ 32\,\ \'B\'\:\ 16\,\ \'A\'\:\ 32\,\ \'S\'\:\ 1\,\ \'D\'\:\ 8\}16384x512_ddr5.json
#configs/x16\{\'C\'\:\ 10\,\ \'R\'\:\ 64\,\ \'B\'\:\ 16\,\ \'A\'\:\ 16\,\ \'S\'\:\ 1\,\ \'D\'\:\ 8\}16384x512.json 
#configs/x16{'C': 8, 'R': 64, 'B': 16, 'A': 16, 'S': 1, 'D': 8}4096x1024.json
#--M 1024 --K 12288 --N 12288
#--M 2048 --K 24576 --N 24576