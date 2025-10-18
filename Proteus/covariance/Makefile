CC = gcc
CFLAGS = -fopenmp -lm -O3 -w 

.PHONY: all

all: covariance_pim.exe covariance_baseline.exe

covariance_pim.exe: covariance.c polybench.c ../util/bbop_manager.c
	$(CC) $^ -DPAPER_DATASET -DRUN_PIM -o $@ $(CFLAGS)

covariance_baseline.exe: covariance.c polybench.c ../util/bbop_manager.c
	$(CC) $^ -DGET_TIME -DPAPER_DATASET -DRUN_BASELINE -o $@ $(CFLAGS)

clean:
	rm -f covariance_pim.exe covariance_baseline.exe
