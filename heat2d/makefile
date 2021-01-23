C=nvcc
NVCCFLAGS= -std=c++11 -arch=sm_60 

all:

heat: heat2D.o
	${C} ${NVCCFLAGS} -o heat2D heat_test.cu heat2D.o

heat2D.o:
	${C} ${NVCCFLAGS} -c heat2D.cu

batch:
	make clean
	sbatch slurm_heat.sh

clean:
	rm -f *.o *.log *.dat heat2D

log:
	vi heat2D.log
