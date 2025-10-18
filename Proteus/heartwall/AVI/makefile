CC = gcc

all: avimod.o avilib.o

# compile supporting function file into object (binary)
avimod.o: avilib.h avimod.c 
	gcc-15  -O3 -g -march=native -fno-unroll-loops -g -c avimod.c

# compile supporting function file into object (binary)
avilib.o: avilib.h avilib.c
	gcc-15  -O3 -g -march=native -fno-unroll-loops -g -c avilib.c

# delete files
clean:
	rm *.o *.out
