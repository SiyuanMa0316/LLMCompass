#!/bin/bash
# config="configs/x16{'C': 10, 'R': 32, 'B': 16, 'A': 16, 'S': 1, 'D': 8}16384x1024.json"
# config="configs/x16{'C': 8, 'R': 64, 'B': 16, 'A': 16, 'S': 1, 'D': 8}4096x1024.json"
cd ../..
config="configs/x16_base_config.json"
precision="int8"
echo "Using config: $config"
# Run GEMM tests
echo "Running GEMM tests..."
echo "Test 1: M=1024, K=12288, N=12288"
python -m experiments.test_gemm_ddr5_multiprocess --config "$config" --M 1024 --K 12288 --N 12288 --precision "$precision" --roofline
echo "Test 2: M=2048, K=24576, N=24576"
python -m experiments.test_gemm_ddr5_multiprocess --config "$config" --M 2048 --K 24576 --N 24576 --precision "$precision" --roofline
echo "Test 3: M=1, K=12288, N=12288"
python -m experiments.test_gemm_ddr5_multiprocess --config "$config" --M 1 --K 12288 --N 12288 --precision "$precision" --roofline
echo "Test 4: M=1, K=24576, N=24576"
python -m experiments.test_gemm_ddr5_multiprocess --config "$config" --M 1 --K 24576 --N 24576 --precision "$precision" --roofline
# Run LLM tests
echo "Running LLM tests..."
echo "Testing GPT-3 175B model prefill"
python -m experiments.test_llm --config "$config" --model llm_model/gpt3-175b.json --precision "$precision" --prefill
echo "Testing GPT-3 175B model decode"
python -m experiments.test_llm --config "$config" --model llm_model/gpt3-175b.json --precision "$precision" --decode
echo "Testing GPT-3 6.7B model prefill"
python -m experiments.test_llm --config "$config" --model LLM_hyper/gpt3-6.7b.json --precision "$precision" --prefill
echo "Testing GPT-3 6.7B model decode"
python -m experiments.test_llm --config "$config" --model LLM_hyper/gpt3-6.7b.json --precision "$precision" --decode
echo "Testing LLaMA 3.1 70B model prefill"
python -m experiments.test_llm --config "$config" --model llm_model/llama-3.1-70b.json --precision "$precision" --prefill
echo "Testing LLaMA 3.1 70B model decode"
python -m experiments.test_llm --config "$config" --model llm_model/llama-3.1-70b.json --precision "$precision" --decode
echo "Testing LLaMA 3.1 8B model prefill"
python -m experiments.test_llm --config "$config" --model llm_model/llama-3.1-8b.json --precision "$precision" --prefill
echo "Testing LLaMA 3.1 8B model decode"
python -m experiments.test_llm --config "$config" --model llm_model/llama-3.1-8b.json  --precision "$precision" --decode
cd -
