/*
---------------------------------------------------------------------------------------
File: matbenchmark.cu
Author: Bryan Garcia
Assignment: HW 2
Course: AM 148 - GPU Programming

Description:
    Benchmarking Matrix operations via CPU and GPU methods.
    Contrasts times associated with CPU exection, GPU 'naive' execution,
    and GPU shared-memory execution.

---------------------------------------------------------------------------------------
*/

#include <vector>
#include "mat.h"
#include "matopers.h"

int main() {

    // Flags for usage in benchmarking.
    int Serial = 0;
    int Naive = 1;
    int Shared = 2;

    // Fixing M = 1 for 'vector' objects.
    int M = 1;

    // Setting some dimension sizes to test.
    vector<int> test_values {16, 128, 1024, 2048, 8192};
    vector<int> mode_values {Serial, Naive, Shared};


    cout << "\n-------------------------------------------------------------------\n";
    cout << "* Beginning Matrix-Vector benchmark *\n";
    cout << "-------------------------------------------------------------------\n";
    for (int N : test_values) {

        cout << "\n===================================================================\n";

        for (int mode : mode_values) {
            // A = N x N matrix.
            Matrix A(N, N, N, MAT_CPU);

            // B = N x 1 vector.
            Matrix B(N, M, M, MAT_CPU);

            // C = N x 1 resultant vector
            Matrix C(A.height, B.width, B.stride, MAT_CPU);

            // Intializing A and B to identity.
            A.ones();
            B.ones();

            cout << "// Matrix A: (Height, Width) = ";
            cout << "(" << A.height << ", " << A.width << ")\n";
            cout << "// Vector B: (Height, Width) = ";
            cout << "(" << B.height << ", " << B.width << ")\n"; 
            cout << "// Vector C: (Height, Width) = ";
            cout << "(" << C.height << ", " << C.width << ")\n"; 
            cout << endl << endl;

            // Benchmark Serial Matrix-Vector multiplication.
            // Benchmark 'Naive' Matrix-Vector kernel.
            // Benchmark shared-memory Matrix-Vector kernel.
            MatMult(A, B, C, mode);
            
            cout << endl << endl;

            A.dealloc();
            B.dealloc();
            C.dealloc();
        }
        cout << "===================================================================\n";
    }

    cout << "\n-------------------------------------------------------------------\n";
    cout << "* Beginning Matrix-Transpose benchmark *\n";
    cout << "-------------------------------------------------------------------\n";
    for (int N : test_values) {
        cout << "\n===================================================================\n";
        for (int mode : mode_values) {

            // A = N x N matrix.
            Matrix A(N, N, N, MAT_CPU);
            cout << "// Matrix A: (Height, Width) = ";
            cout << "(" << A.height << ", " << A.width << ")\n";
            cout << endl << endl;

            // Intializing A.
            A.sequential();

            // Benchmark Serial Matrix transpose.
            // Benchmark 'Naive' Matrix transpose.
            // Benchmark shared-memory Matrix transpose.
            MatTranspose(A, mode);

            cout << endl << endl;
            A.dealloc();
        }
        cout << "===================================================================\n";
    }

}
