CC = gcc
CFLAGS = -fopenmp -lm -O3 -w

.PHONY: all

all: 3mm_pim.exe 3mm_pim_int4.exe 3mm_baseline.exe

3mm_pim.exe: 3mm.c polybench.c ../util/bbop_manager.c
	$(CC) $^ -DMINI_DATASET -DRUN_PIM -o $@ $(CFLAGS)

3mm_pim_int4.exe: 3mm.c polybench.c ../util/bbop_manager.c
	$(CC) $^ -DMINI_DATASET -DINT4_DATA_TYPE -DRUN_PIM -o $@ $(CFLAGS)

3mm_baseline.exe: 3mm.c polybench.c ../util/bbop_manager.c
	$(CC) $^ -DGET_TIME -DMINI_DATASET -DRUN_BASELINE -o $@ $(CFLAGS)

clean:
	rm -f 3mm_pim.exe 3mm_pim_int4.exe 3mm_baseline.exe
