#!/bin/bash
# python -m experiments.test_gemm_gpu --config configs/H100x1_int8.json --M 1 --K 12288 --N 12288
# python -m experiments.test_gemm_gpu --config configs/H100x1_sxm5_int8.json --M 32 --K 32 --N 32 --precision int8
python -m experiments.test_gemm_gpu --config configs/GA100.json --M 1 --K 4096 --N 4096 --precision fp16