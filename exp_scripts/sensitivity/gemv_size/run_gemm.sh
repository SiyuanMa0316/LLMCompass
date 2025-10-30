config="configs/x16_base_config.json"
precision="int8"
echo "Using config: $config"
# Run GEMM tests
echo "Running GEMM tests..."
cd ../../..
for mn_size in 2048 8192 32768; do
    for k_size in 2048 8192 32768; do
        echo "Testing M=1, K=${k_size}, N=${mn_size}"
        python -m experiments.test_gemm_ddr5_multiprocess --config "$config" --M 1 --K "$k_size" --N "$mn_size" --precision "$precision" --roofline
    done
done
cd -
# echo "Test 1: M=1024, K=12288, N=12288"
# python -m experiments.test_gemm_ddr5_multiprocess --config "$config" --M 1024 --K 12288 --N 12288 --precision "$precision" --roofline