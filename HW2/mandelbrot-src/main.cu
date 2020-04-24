/*------------------------------------------------------------------------
    Filename: main.cu
    Programmer: Bryan Garcia
    Course: AM 148

    Description:
        Driver code for the benchmarking of Mandelbrot set generation.
        A serial and parallel algorithm are to be compared.
------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mandelbrot.h"

using namespace std;

void benchmark_mandelbrot(const int width, const int height, const int max_iter, bool serial_mode = true) {
/*--------------------------------------------------------------------------------------
    benchmark_mandelbrot

        Description: Helper function for benchmarking Mandelbrot rendering.

        Parameters:

            width : const int
                Width of image.

            height : const int
                Height of image.

            max_iter : const int
                Escape value for Mandelbrot set generation. 

            serial_mode : bool (optional)
                Flag for running rendering in serial or parallel

--------------------------------------------------------------------------------------*/
    float duration = mandelbrot(width, height, max_iter, serial_mode);
    
    printf("%s", "--------------------------------\n");
    printf("%s Mandelbrot generation: \n", serial_mode ? "SERIAL" : "PARALLEL");
    printf("Width: %d \n", width);
    printf("Height: %d \n", height);
    printf("Max_iter: %d \n", max_iter);
    printf("TOTAL TIME (ms): %f\n", duration);
    printf("%s", "--------------------------------\n\n");
}

int main() {

    const int max_iter = 256;

    printf("%s", "Beginning benchmark...\n");

    // Alternating between serial and parallel rendering...
    benchmark_mandelbrot(1024, 1024, max_iter, true);
    benchmark_mandelbrot(1024, 1024, max_iter, false);

    benchmark_mandelbrot(2048, 2048, max_iter, true);
    benchmark_mandelbrot(2048, 2048, max_iter, false);

    benchmark_mandelbrot(4096, 4096, max_iter, true);
    benchmark_mandelbrot(4096, 4096, max_iter, false);

    benchmark_mandelbrot(8192, 8192, max_iter, true);
    benchmark_mandelbrot(8192, 8192, max_iter, false);

    return 0;
}
