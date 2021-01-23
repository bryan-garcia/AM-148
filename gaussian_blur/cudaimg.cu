#include "cudaimg.h"

__global__
void NaiveGaussianBlurKernel(unsigned char* input_data, unsigned char* output_data, const int width, const int height) {

    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int channel = threadIdx.z + blockIdx.z * blockDim.z;

    if (col >= width or row >= height) return;

    float sum;


    if (col == 0 and row == 0) {
        sum = input_data[(height * channel + row) * width + col] * GB_MASK[1][1] \
        + input_data[(height * channel + row) * width + col + 1] * GB_MASK[1][2] \
        + input_data[(height * channel + row + 1) * width + col] * GB_MASK[2][1] \
        + input_data[(height * channel + row + 1) * width + col + 1] * GB_MASK[2][2];
    }

    else if (col == width - 1 and row == 0) {
        sum = input_data[(height * channel + row) * width + col] * GB_MASK[1][1] \
        + input_data[(height * channel + row) * width + col - 1] * GB_MASK[1][0] \
        + input_data[(height * channel + row + 1) * width + col] * GB_MASK[2][1] \
        + input_data[(height * channel + row + 1) * width + col - 1] * GB_MASK[2][1];
    }

    else if (col == 0 and row == height - 1) {
        sum = input_data[(height * channel + row) * width + col] * GB_MASK[1][1] \
        + input_data[(height * channel + row) * width + col + 1] * GB_MASK[1][2] \
        + input_data[(height * channel + row - 1) * width + col] * GB_MASK[0][1]  \
        + input_data[(height * channel + row - 1) * width + col + 1] * GB_MASK[0][2];
    }

    else if (col == width - 1 and row == height - 1) {
        sum = input_data[(height * channel + row) * width + col] * GB_MASK[1][1] \
        + input_data[(height * channel + row) * width + col - 1] * GB_MASK[1][0] \
        + input_data[(height * channel + row - 1) * width + col] * GB_MASK[0][1] \
        + input_data[(height * channel + row - 1) * width + col - 1] * GB_MASK[0][0];
    }

    // First column excluding corners.
    else if (col == 0) {
        sum = input_data[(height * channel + row - 1) * width + col] * GB_MASK[0][1] \
        + input_data[(height * channel + row - 1) * width + col + 1] * GB_MASK[0][2] \
        + input_data[(height * channel + row) * width + col] * GB_MASK[1][1] \
        + input_data[(height * channel + row) * width + col + 1] * GB_MASK[1][2] \
        + input_data[(height * channel + row + 1) * width + col] * GB_MASK[2][1] \
        + input_data[(height * channel + row + 1) * width + col + 1] * GB_MASK[2][2];
    }

    // Last column excluding corners
    else if (col == width - 1) {
        sum = input_data[(height * channel + row - 1) * width + col] * GB_MASK[0][1] \
        + input_data[(height * channel + row - 1) * width + col - 1] * GB_MASK[0][0] \
        + input_data[(height * channel + row) * width + col] * GB_MASK[1][1] \
        + input_data[(height * channel + row) * width + col - 1] * GB_MASK[1][0] \
        + input_data[(height * channel + row + 1) * width + col] * GB_MASK[2][1] \
        + input_data[(height * channel + row + 1) * width + col - 1] * GB_MASK[2][0];
    } 

    // // First row excluding corners
    else if (row == 0) {
        sum = input_data[(height * channel + row) * width + col - 1] * GB_MASK[1][0] \
        + input_data[(height * channel + row) * width + col] * GB_MASK[1][1] \
        + input_data[(height * channel + row) * width + col + 1] * GB_MASK[1][2] \
        + input_data[(height * channel + row + 1) * width + col - 1] * GB_MASK[2][0] \
        + input_data[(height * channel + row + 1) * width + col] * GB_MASK[2][1] \
        + input_data[(height * channel + row + 1) * width + col + 1] * GB_MASK[2][2];
    }

    // Last row excluding corners
    else if (row == height - 1) {
        sum = input_data[(height * channel + row - 1) * width + col - 1] * GB_MASK[0][0] \
        + input_data[(height * channel + row - 1) * width + col] * GB_MASK[0][1] \
        + input_data[(height * channel + row - 1) * width + col + 1] * GB_MASK[0][2] \
        + input_data[(height * channel + row) * width + col - 1] * GB_MASK[1][0] \
        + input_data[(height * channel + row) * width + col] * GB_MASK[1][1] \
        + input_data[(height * channel + row) * width + col + 1] * GB_MASK[1][2];
    }

    // // Apply full 3x3 mask
    else {
        sum = input_data[(height * channel + row - 1) * width + col - 1] * GB_MASK[0][0] \
        + input_data[(height * channel + row - 1) * width + col] * GB_MASK[0][1] \
        + input_data[(height * channel + row - 1) * width + col + 1] * GB_MASK[0][2] \
        + input_data[(height * channel + row) * width + col - 1] * GB_MASK[1][0] \
        + input_data[(height * channel + row) * width + col] * GB_MASK[1][1] \
        + input_data[(height * channel + row) * width + col + 1] * GB_MASK[1][2] \
        + input_data[(height * channel + row + 1) * width + col - 1] * GB_MASK[2][0] \
        + input_data[(height * channel + row + 1) * width + col] * GB_MASK[2][1] \
        + input_data[(height * channel + row + 1) * width + col + 1] * GB_MASK[2][2];
    }

    sum = sum > 255 ? 255 : sum;
    output_data[(height * channel + row) * width + col] = (unsigned char) sum;
    
}

__global__
void SharedGaussianBlurKernel(unsigned char* input_data, unsigned char* output_data, const int width, const int height) {

    __shared__ unsigned char sharedbuff[3][BLOCK_W*BLOCK_H];

    int col = blockIdx.x * TILE_W + threadIdx.x - R;
    int row = blockIdx.y * TILE_H + threadIdx.y - R;
    int channel = blockIdx.z;

    col = max(0, col); 
    col = min(col, width - 1);
    row = max(0, row);
    row = min(row, height - 1);

    unsigned int kernel_index = (height * channel + row) * width + col;
    unsigned int tile_index = threadIdx.y * BLOCK_H + threadIdx.x;
    sharedbuff[channel][tile_index] = input_data[kernel_index];
    __syncthreads();

    if ( (threadIdx.x >= R) and (threadIdx.x < (BLOCK_W - R)) and (threadIdx.y >= R) and (threadIdx.y < (BLOCK_H - R)) ) {

        float sum = 0;
        for (int i = -R; i <= R; i++) {
            for (int j = -R; j <= R; j++) {
                sum += GB_MASK[i + R][j + R] * sharedbuff[channel][tile_index + (i * blockDim.x) + j];
            }
            __syncthreads();
        }

        sum = floorf(sum);
        sum = sum > 255 ? 255 : sum;
        output_data[kernel_index] = (unsigned char) sum;

    }

}


unsigned char* NaiveGaussianBlur(CImg<unsigned char>& input_img) { 
    
    unsigned char* d_src;
    unsigned char* d_dest;

    unsigned char* h_img;

    float duration;
    size_t buffer_size = input_img.size();

    int width = input_img.width();
    int height = input_img.height();
    int depth = input_img.depth();
    int spectrum = input_img.spectrum();

    // Malloc space for image buffer on device.
    cudaMalloc((void**) &d_src, buffer_size);
    cudaMalloc((void**) &d_dest, buffer_size);

    h_img = new unsigned char[buffer_size];

    // Copy input image data to device.
    cudaMemcpy(d_src, input_img.data(), buffer_size, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_W, BLOCK_H, 1);
    dim3 gridDim((width + TILE_W - 1) / TILE_W, (height + TILE_H - 1) / TILE_H, 3);

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    NaiveGaussianBlurKernel<<<gridDim, blockDim>>>(d_src, d_dest, width, height);
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&duration, start, stop);

    cudaDeviceSynchronize();

    cudaMemcpy(h_img, d_dest, buffer_size, cudaMemcpyDeviceToHost);

    cudaFree(d_src);
    cudaFree(d_dest);

    std::cout << "NaiveGaussianBlur took " << duration << " (ms)..." << std::endl;

    return h_img;


    // Copy image data to buffer on device.
}


unsigned char* SharedGaussianBlur(CImg<unsigned char>& input_img) { 
    
    unsigned char* d_src;
    unsigned char* d_dest;

    unsigned char* h_img;

    float duration;
    size_t buffer_size = input_img.size();

    int width = input_img.width();
    int height = input_img.height();
    int depth = input_img.depth();
    int spectrum = input_img.spectrum();

    // Malloc space for image buffer on device.
    cudaMalloc((void**) &d_src, buffer_size);
    cudaMalloc((void**) &d_dest, buffer_size);

    h_img = new unsigned char[buffer_size];

    // Copy input image data to device.
    cudaMemcpy(d_src, input_img.data(), buffer_size, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_W, BLOCK_H, 1);
    dim3 gridDim((width + TILE_W - 1) / TILE_W, (height + TILE_H - 1) / TILE_H, 3);

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    SharedGaussianBlurKernel<<<gridDim, blockDim>>>(d_src, d_dest, width, height);
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&duration, start, stop);

    cudaDeviceSynchronize();

    cudaMemcpy(h_img, d_dest, buffer_size, cudaMemcpyDeviceToHost);

    cudaFree(d_src);
    cudaFree(d_dest);

    std::cout << "SharedGaussianBlur took " << duration << " (ms)..." << std::endl;

    return h_img;


    // Copy image data to buffer on device.
}