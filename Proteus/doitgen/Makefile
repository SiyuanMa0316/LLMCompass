CC = gcc
CFLAGS = -fopenmp -lm -O3 -w 

.PHONY: all

all: doitgen_pim.exe doitgen_baseline.exe

doitgen_pim.exe: doitgen.c polybench.c ../util/bbop_manager.c
	$(CC) $^ -DPAPER_DATASET -DRUN_PIM -o $@ $(CFLAGS)

doitgen_baseline.exe: doitgen.c polybench.c ../util/bbop_manager.c
	$(CC) $^ -DGET_TIME -DPAPER_DATASET -DRUN_BASELINE -o $@ $(CFLAGS)

clean:
	rm -f doitgen_pim.exe doitgen_baseline.exe
