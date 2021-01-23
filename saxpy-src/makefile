CC=nvcc
FLAGS=-std=c++11

all: main.cu saxpy.o
	$(CC) $(FLAGS) -o saxpy main.cu saxpy.o

saxpy.o: saxpy.cu
	$(CC) $(FLAGS) -c saxpy.cu

clean:
	rm -f *.o *.log

batch:
	sbatch slurm-saxpy.sh
