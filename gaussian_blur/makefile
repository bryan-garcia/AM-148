C=nvcc
NVCCFLAGS =  -arch=sm_60 
CFLAGS = -lX11 -lpthread

img: cudaimg.o
	$(C) $(NVCCFLAGS) $(CFLAGS) -o gauss main.cu cudaimg.o

cudaimg.o:
	$(C) $(NVCCFLAGS) $(CFLAGS) -c cudaimg.cu

clean:
	rm -f gauss.log gauss *.o

#cudaimg.o:
#	nvcc -c cudaimg.cu -LCImg/ -lm -lpthread -lX11
