CC = gcc
CFLAGS = -fopenmp -lm -O3 -w 

.PHONY: all

all: gemm_pim.exe gemm_pim_int4.exe gemm_baseline.exe

gemm_pim.exe: gemm.c polybench.c ../util/bbop_manager.c
	$(CC) $^ -DPAPER_DATASET -DRUN_PIM -o $@ $(CFLAGS)

gemm_pim_int4.exe: gemm.c polybench.c ../util/bbop_manager.c
	$(CC) $^ -DPAPER_DATASET -DINT4_DATA_TYPE -DRUN_PIM -o $@ $(CFLAGS)

gemm_baseline.exe: gemm.c polybench.c ../util/bbop_manager.c
	$(CC) $^ -DGET_TIME -DPAPER_DATASET -DRUN_BASELINE -o $@ $(CFLAGS)

clean:
	rm -f gemm_pim.exe gemm_pim_int4.exe gemm_baseline.exe
