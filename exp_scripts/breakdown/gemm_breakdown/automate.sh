python extract_gemm_ablation.py \
  -i base=run_gemm_output no_popcount=run_gemm_output_no_popcount no_broadcast=run_gemm_output_no_broadcast no_buffer=run_gemm_output_no_buffer \
  -o gemm_ablation.csv

python extract_gemm_ablation.py \
  -i base=run_gemv_output no_popcount=run_gemv_output_no_popcount no_broadcast=run_gemv_output_no_broadcast no_buffer=run_gemv_output_no_buffer \
  -o gemv_ablation.csv

python plot.py --input gemm_ablation.csv --output gemm_ablation_breakdown.png
python plot.py --input gemv_ablation.csv --output gemv_ablation_breakdown.png