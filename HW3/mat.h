/*
---------------------------------------------------------------------------------------
File: mat.h
Author: Steven Reeves, Bryan Garcia
Assignment: HW 2
Course: AM 148 - GPU Programming
Source: https://github.com/stevenireeves/ams148/blob/master/matmul/mat.h

Description:
    Header class for stevenireeves' Matrix helper class. 
    Edits include helper functions and preprocessor directives.
---------------------------------------------------------------------------------------
*/
#ifndef _MAT_H_
#define _MAT_H_


// Constants determining whether on CPU or GPU.
#define MAT_CPU 0
#define MAT_GPU -1

// Constants for loading of Matrix.
#define TO_DEV 1
#define TO_HOST 2

class Matrix {

    public:
        int height;
        int width;
        int stride;
        int type;
        double* data;

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
    __host__
    Matrix(const int height, const int width, const int stride, const int type = MAT_CPU);

    /*
    -----------------------------------------------------------
    copy

        Description: Returns a copy of the calling Matrix.

        Parameters: None

    -----------------------------------------------------------
    */
    Matrix copy();

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
    void load(const Matrix& input_matrix, const int type);

    /*
    -----------------------------------------------------------
    dealloc

        Description: Deallocate memory for Matrix object.

        Parameters: None

    -----------------------------------------------------------
    */
    void dealloc();

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
    void ones(const double c = 1.0);
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
    void sequential(double initial_value = 1.0);

};

#endif