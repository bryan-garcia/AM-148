/*-------------------------------------------------------------------
File: saxpy.h
Programmer: Bryan Garcia
Course: AM 148 - GPU Programming

Description: 
    Prototypes for SAXPY benchmark.
-------------------------------------------------------------------*/

#ifndef _SAXPY_H_
#define _SAXPY_H_

#include <iostream>
#include <chrono>

// Serial SAXPY.
void saxpy_serial(const int dim, float a, float* x, float* y);

// Parallel SAXPY.
__global__
void saxpy_gpu(const int dim, float a, float* x, float* y);

// Benchmark function for SAXPY.
void benchmark_saxpy(const int dim, float a, float* x, float* y, bool serial_mode);

#endif