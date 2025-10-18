all: heartwall_baseline.exe heartwall_pim.exe

heartwall_pim.exe: main.c define.c kernel.c ./AVI/avilib.o ./AVI/avimod.o ../util/bbop_manager.c
	gcc-15 main.c ../util/bbop_manager.c ./AVI/avilib.o ./AVI/avimod.o -I./AVI -lm -fopenmp -o heartwall_pim.exe -DRUN_PIM -O3 -g0 -w -march=native -fno-unroll-loops -DGET_TIME_FUNCTION

heartwall_baseline.exe: main.c define.c kernel.c ./AVI/avilib.o ./AVI/avimod.o
	gcc-15 main.c ./AVI/avilib.o ./AVI/avimod.o -I./AVI -lm -fopenmp -o heartwall_baseline.exe -DRUN_BASELINE -O3 -g0 -w -march=native -fno-unroll-loops -DGET_TIME_FUNCTION

./AVI/avilib.o ./AVI/avimod.o:
	cd AVI; make;

clean:
	rm -f *.o *~ heartwall heartwall_cuda.linkinfo heartwall_baseline.exe heartwall_pim.exe	
