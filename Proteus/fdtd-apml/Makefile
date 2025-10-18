CC = gcc
CFLAGS = -fopenmp -lm -O3 -w 

.PHONY: all

all: fdtd-apml_pim.exe fdtd-apml_baseline.exe

fdtd-apml_pim.exe: fdtd-apml.c polybench.c ../util/bbop_manager.c
	$(CC) $^ -DPAPER_DATASET -DRUN_PIM -o $@ $(CFLAGS)

fdtd-apml_baseline.exe: fdtd-apml.c polybench.c ../util/bbop_manager.c
	$(CC) $^ -DGET_TIME -DPAPER_DATASET -DRUN_BASELINE -o $@ $(CFLAGS)

clean:
	rm -f fdtd-apml_pim.exe fdtd-apml_baseline.exe
