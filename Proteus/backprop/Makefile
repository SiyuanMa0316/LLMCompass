# C compiler
CC = gcc
CC_FLAGS_BASELINE = -g0 -fopenmp -O3 -march=native -w -DGET_TIME_FUNCTION -DGET_TIME -DOPEN 
CC_FLAGS_PIM = -g0 -fopenmp -O3 -march=native -w -DRUN_PIM -DOPEN 

CC_FLAGS = -g0 -fopenmp -O3 -march=native -w 

all: backprop_baseline.exe backprop_pim.exe

backprop_baseline.exe: backprop.c facetrain.c imagenet.c backprop_kernel.c 
	$(CC) $(CC_FLAGS_BASELINE) backprop.c facetrain.c imagenet.c backprop_kernel.c -o backprop_baseline.exe -lm 

backprop_pim.exe: backprop.c facetrain.c imagenet.c backprop_kernel.c ../util/bbop_manager.c
	$(CC) $(CC_FLAGS_PIM) backprop.c facetrain.c imagenet.c backprop_kernel.c ../util/bbop_manager.c -o backprop_pim.exe -lm 

clean:
	rm -f *.o *~ backprop backprop_cuda.linkinfo backprop_baseline.exe backprop_pim.exe
