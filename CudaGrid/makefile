C=nvcc
FLAGS=-std=c++11 -rdc=true
SHARED=--compiler-options '-fPIC'

all:

gbatch:
	sbatch slurm-gtest.sh
	
ebatch:
	sbatch slurm-etest.sh

pbatch:
	sbatch slurm-ptest.sh

gtest: psuedo_ND.o
	${C} ${FLAGS} -o gtest gtest.cu psuedo_ND.o

etest: psuedo_ND.o CudaGrid.o
	${C} ${FLAGS} -o etest CudaGrid_TEST.cu psuedo_ND.o CudaGrid.o

CudaGrid.o: psuedo_ND.o
	${C} ${FLAGS} -c CudaGrid.cu

psuedo_ND.o:
	${C} ${FLAGS} -c psuedo_ND.cu

CudaGrid_Driver.o: 
	${C} ${FLAGS} -c CudaGrid_Driver.cu

lib:
	${C} ${FLAGS} ${SHARED} -c CudaGrid_Driver.cu psuedo_ND.cu CudaGrid.cu
	${C} -shared -o lib.so *.o

clean:
	rm -f *.o gtest etest *.log lib.so