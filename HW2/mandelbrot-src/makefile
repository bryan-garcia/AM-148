CC=nvcc

all: mandelbrot

mandelbrot: main.cu mandelbrot.o bmp.o
	$(CC) -o mandelbrot main.cu mandelbrot.o bmp.o

mandelbrot.o: mandelbrot.cu mandelbrot.h
	$(CC) -c mandelbrot.cu

bmp.o: bmp.c bmp.h
	$(CC) -c bmp.c

batch:
	sbatch slurm-mandelbrot.sh
clean:
	rm -f mandelbrot.bmp *.o *.out
