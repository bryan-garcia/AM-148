/*------------------------------------------------------------------------
    Filename: mandelbrot.h
    Programmer: Bryan Garcia
    Course: AM 148

    Description:
        Prototypes for rendering in serial and parallel.
------------------------------------------------------------------------*/
#ifndef _MANDELBROT_H_
#define _MANDELBROT_H_

extern "C" {
    #include "bmp.h"
}

// Serial Mandelbrot set generation.
void render(char* output_buffer, const int width, const int height, const int max_iter);

// Parallel Mandelbrot set generation.
__global__ void render_cuda(char* device_buffer, const int width, const int height, const int max_iter);

// Wrapper for Mandelbrot set rendering.
float mandelbrot(const int width, const int height, const int max_iter, bool serial_run);


#endif
