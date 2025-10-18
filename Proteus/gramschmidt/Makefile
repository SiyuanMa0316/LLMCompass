CC = gcc
CFLAGS = -fopenmp -lm -O3 -w

.PHONY: all

all: gramschmidt_pim.exe gramschmidt_baseline.exe

gramschmidt_pim.exe: gramschmidt.c polybench.c ../util/bbop_manager.c
	$(CC) $^ -DPAPER_DATASET -DRUN_PIM -o $@ $(CFLAGS)

gramschmidt_baseline.exe: gramschmidt.c polybench.c ../util/bbop_manager.c
	$(CC) $^ -DGET_TIME -DPAPER_DATASET -DRUN_BASELINE -o $@ $(CFLAGS)

clean:
	rm -f gramschmidt_pim.exe gramschmidt_baseline.exe
