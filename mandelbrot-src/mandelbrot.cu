/*------------------------------------------------------------------------
    Filename: mandelbrot.cu
    Programmer: Bryan Garcia
    Course: AM 148

    Description:
        Implementation of functions for rendering of mandelbrot set.
        Both a serial and parallel algorithm are implemented here.
------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>

#include "mandelbrot.h"

void render(char* output_buffer, const int width, const int height, const int max_iter) {
/*--------------------------------------------------------------------------------------
    render

        Description: Generate data and populate a buffer with Mandelbrot
        set. To be used as a serial code.

        Parameters:

            output_buffer : char*
                Buffer to be populated and eventually written as a BMP image.

            width : const int
                Width of image.

            height : const int
                Height of image.

            max_iter : const int
                Escape value for Mandelbrot set generation. 

        Post:
            output_buffer will be modified to contain Mandelbrot data.

--------------------------------------------------------------------------------------*/

    // Traverse height of 'image'.
    for (int i = 0; i < height; i++) {

        // Traverse width of 'image'.
        for (int j = 0; j < width; j++) {
            
            float x = 0.0;
            float y = 0.0;


            // Set an origin.
            float x_origin = 3.25 * ((float) i / height) - 2;
            float y_origin = 2.5 * ((float) j / width) - 1.25;

            unsigned int iteration_count = 0;

            // Checking for escape condition.
            while (x*x + y*y < 4 && iteration_count < max_iter) {
                float x_temp = x*x - y*y + x_origin;
                y = 2 * x * y + y_origin;
                x = x_temp;
                iteration_count++;
            }

            int index = 3 * (width*i + j);

            /* Write to output_buffer in RGB format */

            // Reached max_iter value: set to null.
            if (iteration_count == max_iter) {
                output_buffer[index] = 0;
                output_buffer[index + 1] = 0;
                output_buffer[index + 2] = 0;
            }
            // Reached max_iter value: set to iteration_count intensity.
            else {
                output_buffer[index] = iteration_count;
                output_buffer[index + 1] = iteration_count;
                output_buffer[index + 2] = iteration_count;
            }
        }
    }
}

__global__ 
void render_cuda(char* device_buffer, const int width, const int height, const int max_iter) {
/*--------------------------------------------------------------------------------------
    render_cuda

        Description: Generate data and populate a buffer with Mandelbrot
        set on a GPU. 

        Parameters:

            output_buffer : char*
                Buffer to be populated and eventually written as a BMP image.

            width : const int
                Width of image.

            height : const int
                Height of image.

            max_iter : const int
                Escape value for Mandelbrot set generation. 

        Post:
            output_buffer will be modified to contain Mandelbrot data.

--------------------------------------------------------------------------------------*/


    // Get process ID values for the x and y directions.
    int x_dim = blockIdx.x * blockDim.x + threadIdx.x;
    int y_dim = blockIdx.y * blockDim.y + threadIdx.y;

    int index = 3 * (width * y_dim + x_dim);

    // Setting an origin relative to an ID.
    float x_origin = 3.25 * ((float) x_dim / width) - 2;
    float y_origin = 2.5 * ((float) y_dim / height) - 1.25;

    float x = 0.0;
    float y = 0.0;

    int iteration_count = 0;

    // Check for escape condition.
    while (x*x + y*y <=4 && iteration_count < max_iter) {
        float x_temp = x*x - y*y + x_origin;
        y = 2 * x * y + y_origin;
        x = x_temp;
        iteration_count++;
    }

    /* Write to output_buffer in RGB format */

    // Reached max_iter value: set to null.
    if (iteration_count == max_iter) {
        device_buffer[index] = 0;
        device_buffer[index + 1] = 0;
        device_buffer[index + 2] = 0;
    }
    
    // Reached max_iter value: set to iteration_count intensity.
    else {
        device_buffer[index] = iteration_count;
        device_buffer[index + 1] = iteration_count;
        device_buffer[index + 2] = iteration_count;
    }
}


float mandelbrot(const int width, const int height, const int max_iter, bool serial_mode = true) {
/*--------------------------------------------------------------------------------------
    mandelbrot

        Description: 
            Wrapper for the rendering of a Mandelbrot image. Determines
            whether to perform in serial or parallel. Generated output BMP
            image.

        Parameters:

            width : const int
                Desired width of BMP image.

            height : const int
                Desired height of BMP image.

            max_iter : const int
                Maximum iteration value for rendering of Mandelbrot.

            serial_mode : bool (optional)
                Flag for serial or parallel mode.

        Return:

            duration : float
                Time (in ms) elapsed during rendering.


            Post-condition:
                BMP image of Mandelbrot set with the provided dimensions will
                be written.

--------------------------------------------------------------------------------------*/

    // Calculating size of image to allocate.
    size_t buffer_size = sizeof(char) * width * height * 3;

    // Allocate buffer size.
    char* host_buffer = new char[buffer_size];

    // For benchmark purposes.
    float duration;

    if (not serial_mode) {
        /* Launching rendering in parallel */

        char* device_buffer;
        cudaMalloc((void**) &device_buffer, buffer_size);

        dim3 blockDim(16, 16, 1);
        dim3 gridDim(width / blockDim.x, height / blockDim.y, 1);

        cudaEvent_t start; 
        cudaEvent_t end;

        // Starting timer.
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        cudaEventRecord(start, 0);

        render_cuda<<<gridDim, blockDim>>>(device_buffer, width, height, max_iter);

        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);
        
        cudaDeviveSynchronize()
        cudaEventElapsedTime(&duration, start, end);
        cudaEventDestroy(start);
        cudaEventDestroy(end);

        cudaMemcpy(host_buffer, device_buffer, buffer_size, cudaMemcpyDeviceToHost);
        cudaFree(device_buffer);

    }

    else {
        /* Launching rendering in serial */
        clock_t start, end;
        
        start = clock();
        render(host_buffer, width, height, max_iter);
        end = clock();

        duration = ((float) end - start) / (CLOCKS_PER_SEC / 1000);

    }

    write_bmp("mandelbrot.bmp", width, height, host_buffer);

    delete host_buffer;

    return duration;
}
