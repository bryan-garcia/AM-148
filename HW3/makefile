C=nvcc
NVCCFLAGS = -std=c++11 -rdc=true
# GDB =-g

all: matbenchmark

matbenchmark: matbenchmark.cu mat.o matopers.o
	${C} ${NVCCFLAGS} ${GDB} -o matmult matbenchmark.cu mat.o matopers.o

mat.o: mat.cu
	${C} ${NVCCFLAGS} ${GDB} -dc mat.cu

matopers.o: matopers.cu
	${C} ${NVCCFLAGS} ${GDB} -dc matopers.cu

batch: slurm-matmult.sh
	make clean
	sbatch slurm-matmult.sh

clean:
	rm -f matmult *.o *.log
