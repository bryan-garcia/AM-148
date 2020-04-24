/*-------------------------------------------------------------------
File: main.cu
Programmer: Bryan Garcia
Course: AM 148 - GPU Programming

Description: 
    Benchmarking vector addition with scalar mult. on CPU vs GPU.
    Main.cu serves as the testing suite.
-------------------------------------------------------------------*/

#include "saxpy.h"


// Helper function to free dynamically allocated data.
void free_data(float* x, float* y) {

    delete x;
    delete y;
}


// Main loop: Driver code for benchmarking.
int main() {


    // Preparing data to be utilized for SAXPY.

    // testVals contains the various vector sizes to be
    // benchmarked.
    int testVals[] = {16, 128, 1024, 2048, 65536};
    int N;
    float a = 2.0f;
    float* x;
    float* y;
    bool serial_mode;

    // Looping over all vector sizes listed above.
    for (int i = 0; i < 5; i++) {
        
        N = testVals[i];

        /* Beginning of CPU serial SAXPY benchmark*/
        x = (float*) malloc(N * sizeof(float));
        y = (float*) malloc(N * sizeof(float));

        for (int j = 0; j < N; j++) {
            x[j] = 1.0;
            y[j] = 2.0;

        }
        serial_mode = true;
        benchmark_saxpy(N, a, x, y, serial_mode);
        free_data(x,y);
        /* End of CPU serial SAXPY benchmark*/

        /* Beginning of GPU parallel SAXPY benchmark*/
        x = (float*) malloc(N * sizeof(float));
        y = (float*) malloc(N * sizeof(float));

        for (int j = 0; j < N; j++) {
            x[j] = 1.0;
            y[j] = 2.0;

        }
        serial_mode = false;
        benchmark_saxpy(N, a, x, y, serial_mode);
        free_data(x,y);
        /* End of GPU parallel SAXPY benchmark*/

    }
    
    return 0;
}