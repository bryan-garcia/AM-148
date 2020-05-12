/*-------------------------------------------------------------------
File: saxpy.cu
Programmer: Bryan Garcia
Course: AM 148 - GPU Programming

Description: 
    Implementation of saxpy.h functions.
-------------------------------------------------------------------*/

#include "saxpy.h"

void saxpy_serial(const int dim, float a, float* x, float *y) {
    for (int i = 0; i < dim; i++) 
        y[i] += a * x[i];
}

__global__
void saxpy_gpu(const int dim, float a, float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) y[i] = a * x[i] + y[i];
}

void benchmark_saxpy(const int dim, float a, float* x, float* y, bool serial_mode) {
    /*
        benchmark_saxpy
            
            Description:
                Function for providing benchmarking of SAXPY in either serial or
                parallel.

            Parameters:

                dim : const int
                    Size of incoming vectors to be added.

                a : float
                    Scalar coefficient of vector x.

                x : float*
                    Array of floats to be added to y.

                y : float* 
                    Array of floats to be added to x.

                serial_mode : bool
                    Boolean flag for determining which benchmark method to utilize.
                    
                    Note:
                        Serial SAXPY is benchmarked with the C++ chrono library.
                        Parallel SAXPY uses cudaEvents.

    */
    
    size_t buffer_size = dim * sizeof(float);
    float duration;

    // Declaring/initializing here, but will be reset...
    //  (I was running into some issues otherwise...)
    auto startc = std::chrono::high_resolution_clock::now();
    auto endc = std::chrono::high_resolution_clock::now();
    auto chrono_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(endc - startc);

    if (serial_mode) {

        // Begin timing of serial saxpy function.
        startc = std::chrono::high_resolution_clock::now();
        saxpy_serial(dim, a, x, y);
        endc = std::chrono::high_resolution_clock::now();
        chrono_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(endc - startc);
    }

    else {

        // Begin parallel saxpy.
        float* d_x;
        float* d_y;

        cudaMalloc((void**) &d_x, buffer_size);
        cudaMalloc((void**) &d_y, buffer_size);

        cudaMemcpy(d_x, x, buffer_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y, buffer_size, cudaMemcpyHostToDevice);

        cudaEvent_t start; 
        cudaEvent_t end;

        // Start timing kernel...
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        cudaEventRecord(start, 0);
        startc = std::chrono::high_resolution_clock::now();
        saxpy_gpu<<<(dim + 255) / 256, 256>>>(dim, a, d_x, d_y);
        cudaDeviceSynchronize();
        endc = std::chrono::high_resolution_clock::now();
        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);
        cudaDeviceSynchronize();
        cudaEventElapsedTime(&duration, start, end);
        cudaEventDestroy(start);
        cudaEventDestroy(end);
        
        cudaMemcpy(y, d_y, buffer_size, cudaMemcpyDeviceToHost);

        cudaFree(d_x);
        cudaFree(d_y);

        chrono_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(endc - startc);

        std::cout << "\n\n" << chrono_duration.count()  << "\n\n";
    }

    // Prompt reporting Saxpy time (in ns) //
    std::cout << "--------------------------------\n";
    if (serial_mode) std::cout << "SERIAL"; 
    else std::cout << "PARALLEL";
    std::cout << " saxpy: " << std::endl;
    std::cout << "Dimension: " << dim << std::endl;
    std::cout << "TOTAL TIME (ns): ";
    if (serial_mode) std::cout << chrono_duration.count();
    else std::cout << duration * 1000000.0;
    std::cout << "\n--------------------------------\n\n";
}