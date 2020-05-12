/*
---------------------------------------------------------------------------------------
File: mat.cu
Author: Steven Reeves, Bryan Garcia
Assignment: HW 2
Course: AM 148 - GPU Programming
Original-Source: https://github.com/stevenireeves/ams148/blob/master/matmul/mat.cu

Description:
    Definitions for stevenireeves' Matrix helper class. 
---------------------------------------------------------------------------------------
*/

#include "mat.h"

/*
-----------------------------------------------------------
Matrix(...)

    Description: Parameterized constructor for Matrix obj.

    Parameters:

        height : int
            Matrix geometric height.

        width : int
            Matrix geometrix width.

        stride : int
            Data stride value.

        type : int
            Integer flag for indicating whether on host or
            device. (See MAT_CPU and MAT_GPU in mat.h).

-----------------------------------------------------------
*/
Matrix::Matrix(const int height, const int width, const int stride, const int type) {
    this->width = width;
    this->height = height;
    this->stride = stride;
    this->type = type;

    // Malloc is on CPU
    if (type == MAT_CPU) {
        data = new double[width * height];
    }
    // Malloc data on GPU
    else if (type == MAT_GPU) {
        cudaMalloc(&data, width * height * sizeof(double));
    }

}

/*
-----------------------------------------------------------
copy

    Description: Returns a copy of the calling Matrix.

    Parameters: None

-----------------------------------------------------------
*/
Matrix Matrix::copy() {
    Matrix Copy(height, width, stride, type);
    return Copy;
}


/*
-----------------------------------------------------------
load

    Description: Copy data on host, to host, or to device.

    Parameters:

        input_matrix : const Matrix&
            Matrix to copy data from.

        dir : int
            Location/Directions to copy data to. 
            - (MAT_CPU : memcpy on host)
            - (TO_DEV : cudaMemcpy to device)
            - (TO_HOST : cudaMemcpy to host)

-----------------------------------------------------------
*/
void Matrix::load(const Matrix& input_matrix, const int dir) {

    // Determine buffer size to allocate.
    size_t buffer_size = input_matrix.height * input_matrix.width * sizeof(double);

    if (dir == MAT_CPU) {
        memcpy(data, input_matrix.data, buffer_size);
    }
    else if (dir == TO_DEV) {
        cudaMemcpy(data, input_matrix.data, buffer_size, cudaMemcpyHostToDevice);
    }
    else if (dir == TO_HOST) {
        cudaMemcpy(data, input_matrix.data, buffer_size, cudaMemcpyDeviceToHost);
    }
}
/*
-----------------------------------------------------------
dealloc

    Description: Deallocate memory for Matrix object.

    Parameters: None

-----------------------------------------------------------
*/
void Matrix::dealloc() {

    if (type == MAT_CPU)
        delete data;

    else if (type == MAT_GPU)
        cudaFree(data);
}

/*
-----------------------------------------------------------
ones

    Description: Set all 'data' values equal to a scalar
    double. Defaults to 1.0.

    Parameters:

        c : const double (optional)
            Scalar multiple to apply to ones operation.

-----------------------------------------------------------
*/
void Matrix::ones(const double c) {

    if (type == MAT_CPU) {
        for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++) {
                data[i*stride + j] = c * 1.0;
        }
    }
}

/*
-----------------------------------------------------------
sequential

    Description: Set values to be increasing sequentially
    from a value. Default starts at 1.0.

    Parameters:

        start : const double (optional)
            Scalar value to start from.

-----------------------------------------------------------
*/
void Matrix::sequential(double start) {

    if (type == MAT_CPU) {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                data[i*stride + j] = start;
                start += 1.0;
            }
        }
    }
}