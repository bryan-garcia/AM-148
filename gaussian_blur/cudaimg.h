#ifndef _CUDA_IMG_H_
#define _CUDA_IMG_H_

#include "CImg.h"
#include <iostream>
#include <cstdio>
using namespace cimg_library;

// Defining thread block sizes.
//  - Kernel launch will operate on TILE_W * TILE_H pixels
#define TILE_W 16
#define TILE_H 16

// Radius of our stencil. R = 1 corresponds to our 3x3 Gaussian Blur matrix.
#define R 1

// Stencil with cell padding
#define BLOCK_W (TILE_W + (2*R))
#define BLOCK_H (TILE_H + (2*R))

// Gaussian Blur matrix for convolution.

__device__ 
float GB_MASK[3][3] = { 
    {0.0625, 0.125, 0.0625}, 
    {0.125, 0.25, 0.125}, 
    {0.0625, 0.125, 0.0625} 
};

// __device__ 
// float GB_MASK[7][7] = { 
//     {0.00000067,  0.00002292,  0.00019117,  0.00038771,  0.00019117,  0.00002292,  0.00000067},
//     {0.00002292, 0.00078633,  0.00655965,  0.01330373,  0.00655965,  0.00078633,  0.00002292},
//     {0.00019117, 0.00655965,  0.05472157,  0.11098164,  0.05472157,  0.00655965,  0.00019117},
//     {0.00038771, 0.01330373,  0.11098164,  0.22508352,  0.11098164,  0.01330373,  0.00038771},
//     {0.00019117, 0.00655965,  0.05472157,  0.11098164,  0.05472157,  0.00655965,  0.00019117},
//     {0.00002292, 0.00078633,  0.00655965,  0.01330373,  0.00655965,  0.00078633,  0.00002292},
//     {0.00000067,  0.00002292,  0.00019117,  0.00038771,  0.00019117,  0.00002292,  0.00000067}
// };


__global__
void NaiveGaussianBlurKernel(unsigned char* input_data, unsigned char* output_data, const int width, const int height);

__global__
void SharedGaussianBlurKernel(unsigned char* input_data, unsigned char* output_data, const int width, const int height);

unsigned char* NaiveGaussianBlur(CImg<unsigned char>& input_img);

unsigned char* SharedGaussianBlur(CImg<unsigned char>& input_img);
 

#endif