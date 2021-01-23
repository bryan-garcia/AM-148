/*
---------------------------------------------------------------------------------------
File: matopers.h
Author: Steven Reeves, Bryan Garcia
Assignment: HW 2
Course: AM 148 - GPU Programming
Original-Source: https://github.com/stevenireeves/ams148/blob/master/matmul/matmul.h

Description:
    Header for supported Matrix operations. Building off of 'matmul.*' source code.
---------------------------------------------------------------------------------------
*/

#ifndef _MAT_OPERS_H_
#define _MAT_OPERS_H_

#include <iostream>
#include <chrono>
#include "mat.h"
using namespace std;


// Defining block size here...
#define block_size 32

/*-----------------------------------------------------------
printMatrix

    Description: Print matrix in row-column format.

    Parameters:

        prec : int (optional)
            Precision value to provide to cout.precision.

-----------------------------------------------------------*/
void printMatrix(const Matrix& A, int prec = 6);


__global__
void SharedMatVectMultKernel(const Matrix A, const Matrix B, Matrix C);
/*-----------------------------------------------------------
SharedMatVectMultKernel

    Description: Matrix vector multiplication utilizing 
    shared-memory on GPU.

        "C = AB"

    Parameters:

        A : const Matrix
            Input Matrix to multiply.

        B : const Matrix
            Input 'vector' to multiply.

        C : Matrix
            Resultant 'vector' of AB.

-----------------------------------------------------------*/


__global__
void NaiveMatVectMultKernel(const Matrix A, const Matrix B, Matrix C);
/*-----------------------------------------------------------
NaiveMatVectMultKernel

    Description: Matrix vector multiplication utilizing 
    'naive' algorithm on GPU.

        "C = AB"

    Parameters:

        A : const Matrix
            Input Matrix to multiply.

        B : const Matrix
            Input 'vector' to multiply.

        C : Matrix
            Resultant 'vector' of AB.

-----------------------------------------------------------*/


void SerialMatMult(const Matrix A, const Matrix B, Matrix C);
/*-----------------------------------------------------------
SerialMatMultKernel

    Description: Matrix vector multiplication utilizing 
    sequential CPU method.

        "C = AB"

    Parameters:

        A : const Matrix
            Input Matrix to multiply.

        B : const Matrix
            Input 'vector' to multiply.

        C : Matrix
            Resultant 'vector' of AB.

-----------------------------------------------------------*/


void MatMult(const Matrix& A, const Matrix& B, Matrix& C, int mode = MAT_CPU);
/*----------------------------------------------------------------------
MatMult

    Description:
        Wraps serial and GPU implementation of matrix multiplication.
        To be used to demonstrate matrix-vector product.

        C = AB

    Parameters:
        A : const Matrix
            Left hand Matrix A in the above equation.
        B : const Matrix
            Right hand Matrix B in the above equation.
        C : Matrix&
            Resultant Matrix of AB.
        mode : int
            Determines serial or GPU mode;
                - 0 (default) corresponds to serial execution.
                - 1 corresponds 'Naive' global memory GPU execution.
                - 2 corresponds to shared memory GPU execution.

----------------------------------------------------------------------*/


__global__
void SharedMatTransposeKernel(Matrix A, Matrix A_T);
/*-----------------------------------------------------------
SharedMatTransposeKernel

    Description: Matrix transpose operation utilizing 
    shared-memory on GPU.

    Parameters:

        A : Matrix
            Matrix containing data to 'transpose'.

        A_T : Matrix
            Matrix object to copy transposed data into.

-----------------------------------------------------------*/


__global__
void NaiveMatTransposeKernel(Matrix A, Matrix A_T);
/*-----------------------------------------------------------
NaiveMatTransposeKernel

    Description: Matrix transpose operation utilizing 
    'naive' method on GPU.

    Parameters:

        A : Matrix
            Matrix containing data to 'transpose'.

        A_T : Matrix
            Matrix container to copy transposed data into.

-----------------------------------------------------------*/

void SerialMatTranspose(Matrix& A);
/*-----------------------------------------------------------
SerialMatTransposeKernel

    Description: Matrix transpose operation utilizing 
    sequential method on CPU.

    Parameters:

        A : Matrix&
            Matrix to transpose. Modifies A to contain data
            in transposed format.

-----------------------------------------------------------*/

/*----------------------------------------------------------------------
MatTranspose

    Description:
        Wraps serial and GPU implementation of matrix transpose.

        C = AB

    Parameters:
        A : Matrix&
            Matrix A to perform transpose on
        B : const Matrix
            Right hand Matrix B in the above equation.
        C : Matrix&
            Resultant Matrix of AB.
        mode : int
            Determines serial or GPU mode;
                - 0 (default) corresponds to serial execution.
                - 1 corresponds 'Naive' global memory GPU execution.
                - 2 corresponds to shared memory GPU execution.

----------------------------------------------------------------------*/
void MatTranspose(Matrix& A, int mode = MAT_CPU);

#endif
