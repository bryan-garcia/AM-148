/*
---------------------------------------------------------------------------------------
File: matopers.cu
Author: Steven Reeves, Bryan Garcia
Assignment: HW 2
Original-Source: https://github.com/stevenireeves/ams148/blob/master/matmul/matmul.cu

Description:
    Implements serial, 'naive', and shared-memory Matrix operations.
---------------------------------------------------------------------------------------
*/

#include "matopers.h"
#include "submat.h"

void printMatrix(const Matrix& M, int prec) {
/*-----------------------------------------------------------
printMatrix

    Description: Print matrix in row-column format. Used for
    debugging my code.

    Parameters:

        prec : int (optional)
            Precision value to provide to cout.precision.

-----------------------------------------------------------*/

    // If on CPU...
    if (M.type == MAT_CPU) {

        // Setting precision and printing data.
        cout.precision(prec); 
        for (int i = 0; i < M.height; i++) {
            for (int j = 0; j < M.width; j++) {
                cout << fixed << M.data[i*M.stride + j] << " ";
            }
            cout << endl;
        }
        cout << endl;
        cout.precision(0);
    }

    // If on GPU...
    else if (M.type == MAT_GPU) {

        cout << "Fetching Data from GPU...\n";

        // Create a temporary matrix on Host
        Matrix tempMatrix(M.height, M.width, M.stride, MAT_CPU);

        // Load data from GPU into temp Matrix.
        tempMatrix.load(M, TO_HOST);

        // Print as usual
        cout.precision(6);
        for (int i = 0; i < tempMatrix.height; i++) {
            for (int j = 0; j < tempMatrix.width; j++) {
                cout << fixed << tempMatrix.data[i*tempMatrix.stride + j] << " ";
            }
            cout << endl;
        }
        cout << endl;
        cout.precision(0);
    }

}


__global__
void SharedMatVectMultKernel(const Matrix A, const Matrix B, Matrix C) {
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

    // Prepping shared buffer for A Matrix
    __shared__ double As[block_size][block_size];

    // Prepping shared buffer for B Matrix (column vector).
    __shared__ double Bs[block_size * block_size][1];

    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    int row = threadIdx.y;
    int col = threadIdx.x;

    double cvalue = 0.0;

    // Determine how many times we should apply our block to A.
    for (int i = 0; i < (A.width - 1) / block_size + 1; i++) {
        subMatrix _A(A, block_size, blockRow, i);
        subMatrix _B(B, block_size, i, blockCol);

        // Loading data into shared memory
        As[row][col] = _A.GetElem(row, col);
        Bs[row][0] = _B.GetElem(row, 0);

        // Wait! 
        __syncthreads();

        for (int j = 0; j < block_size; j++) {
            cvalue += As[row][j] * Bs[j][0];
        }

        // Wait!
        __syncthreads();
    }

    // Load into C
    subMatrix _C(C, block_size, blockRow, blockCol);
    _C.SetElem(row, col, cvalue);
}


__global__
void NaiveMatVectMultKernel(const Matrix A, const Matrix B, Matrix C) {
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

    // Get process ID
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    double cvalue = 0.0;

    if (row < A.height and col < B.width) {
        for (int i = 0; i < A.width; i++) {
            cvalue += A.data[row * A.stride + i ] * B.data[i * B.stride + col];
        }
        C.data[row*C.stride + col] = cvalue;
    }
}

void SerialMatMult(const Matrix A, const Matrix B, Matrix C) {
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
    // Running over rows and columns
    for (int i = 0; i < A.height; i++) {
        for (int j = 0; j < B.width; j++) {

            double C_val = 0.0;

            // Performing dot product
            for (int k = 0; k < A.width; k++) {
                C_val += A.data[i * A.stride + k] * B.data[k * B.stride + j];
            }
            C.data[i * C.stride + j] = C_val;
        }
    }
}


void MatMult(const Matrix& A, const Matrix& B, Matrix& C, int mode) {
/*----------------------------------------------------------------------
MatMult

    Description:
        Wraps serial and GPU implementation of matrix multiplication.
        To be used to demonstrate matrix-vector product.

        C = A * B

    Parameters:
        A : const Matrix&
            Left hand Matrix A in the above equation.
        B : const Matrix&
            Right hand Matrix B in the above equation.
        C : Matrix&
            Resultant Matrix of A*B.
        mode : int
            Determines serial or GPU mode;
                - 0 (default) corresponds to serial execution.
                - 1 corresponds 'Naive' global memory GPU execution.
                - 2 corresponds to shared memory GPU execution.

----------------------------------------------------------------------*/

    // Running in CPU mode! 
    if (mode == 0) {

        cout << "// Starting serialMatMult..." << endl;

        // Starting timer...
        auto start = chrono::high_resolution_clock::now();

        SerialMatMult(A, B, C);

        // Ending timer...
        auto stop = chrono::high_resolution_clock::now();

        auto duration = chrono::duration_cast<chrono::nanoseconds>(stop - start);
        cout << "// End of serialMatMult..." << endl;

        cout.precision(6);
        cout << "// serialMatMult completed with an elapsed time of: ";

        // Converting from ns to ms here...
        cout << fixed << duration.count() * 0.000001 << " (ms)." << endl;
        cout.precision(0);

    }
    
    // Running in Naive GPU mode...
    else if (mode == 1) {
        
        // Allocate and load device Matrix d_A
        Matrix d_A(A.height, A.width, A.stride, MAT_GPU);
        d_A.load(A, TO_DEV);

        // Allocate and load device Matrix d_B
        Matrix d_B(B.height, B.width, B.stride, MAT_GPU);
        d_B.load(B, TO_DEV);
        
        // Allocate device Matrix d_C
        Matrix d_C(C.height, C.width, C.stride, MAT_GPU);

        dim3 blockDim(block_size, block_size);
        dim3 gridDim(max(B.width / blockDim.x, 1), max(A.height / blockDim.y, 1));

        cout << "// Launching 'Naive' MatMult Kernel..." << endl;
        cout << "// BlockDim (x,y): " << blockDim.x << " " << blockDim.y << endl;
        cout << "// GridDim (x,y): " << gridDim.x << " " << gridDim.y << endl;

        float duration;
        cudaEvent_t start, stop;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Start timing kernel...
        cudaEventRecord(start, 0);

        NaiveMatVectMultKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C);

        // Stop timing kernel...
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&duration, start, stop);

        cudaDeviceSynchronize();
        cout << "// End of 'Naive' MatMul Kernel..." << endl;

        cout.precision(6);
        cout << "// 'Naive' MatMult completed with an elapsed time of ";
        cout << fixed << duration << " (ms)." << endl;
        cout.precision(0);

        // Make sure to load memory back to host.
        C.load(d_C, TO_HOST);

        // Cleaning up...
        d_A.dealloc();
        d_B.dealloc();
        d_C.dealloc();
    }

    // Running in shared-memory mode...
    else if (mode == 2) {

        cout << "Launching Shared Memory MatMult Kernel..." << endl;

        // Allocate and load device Matrix d_A
        Matrix d_A(A.height, A.width, A.stride, MAT_GPU);
        d_A.load(A, TO_DEV);

        // Allocate and load device Matrix d_B
        Matrix d_B(B.height, B.width, B.stride, MAT_GPU);
        d_B.load(B, TO_DEV);
        
        // Allocate device Matrix d_C
        Matrix d_C(C.height, C.width, C.stride, MAT_GPU);

        dim3 blockDim(block_size, block_size);
        dim3 gridDim(max(B.width / blockDim.x, 1), max(A.height / blockDim.y, 1));

        cout << "// Launching 'Shared' MatMult Kernel..." << endl;
        cout << "// BlockDim (x,y): " << blockDim.x << " " << blockDim.y << endl;
        cout << "// GridDim (x,y): " << gridDim.x << " " << gridDim.y << endl;

        float duration;
        cudaEvent_t start, stop;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Start timing kernel...
        cudaEventRecord(start, 0);
        SharedMatVectMultKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C);

        // Stop timing kernel...
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&duration, start, stop);

        cudaDeviceSynchronize();
        cout << "// End of 'Shared' MatMul Kernel..." << endl;

        cout.precision(6);
        cout << "// 'Shared' MatMult completed with an elapsed time of ";
        cout << fixed << duration << " (ms)." << endl;
        cout.precision(0);

        // Make sure to load memory back to host.
        C.load(d_C, TO_HOST);

        // Cleaning up...
        d_A.dealloc();
        d_B.dealloc();
        d_C.dealloc();

        // Probably could've combined if-else logic on both kernel cases...
    }

    else
        cout << "Unrecognized MatMult 'mode'..." << endl;

    return;
}


__global__
void SharedMatTransposeKernel(Matrix A, Matrix A_T) {
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

    // Prep shared mem buffer...
    __shared__ double ShareLaBeouf [block_size][block_size];

    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    int row = threadIdx.y;
    int col = threadIdx.x;

    if (A.width / block_size) {

        subMatrix _AT(A_T, block_size, blockRow, blockCol);

        subMatrix _A(A, block_size, blockCol, blockRow);

        ShareLaBeouf[row][col] = _A.GetElem(row, col);

        __syncthreads();

        _AT.SetElem(col, row, ShareLaBeouf[row][col]);
    }

    else {

        subMatrix _AT(A_T, block_size, blockRow, blockCol);

        subMatrix _A(A, block_size, blockCol, blockRow);

        if (row < A.height and col < A.width) {
            ShareLaBeouf[row][col] = _A.GetElem(row, col);
            __syncthreads();

            _AT.SetElem(col, row, ShareLaBeouf[row][col]);
        }

    }

}

__global__
void NaiveMatTransposeKernel(Matrix A, Matrix A_T) {
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
    
    // Getting all mah IDs
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    int row = threadIdx.y;
    int col = threadIdx.x;

    subMatrix _A(A, block_size, blockRow, blockCol);
    subMatrix _AT(A_T, block_size, blockCol, blockRow);

    _AT.SetElem(col, row, _A.GetElem(row,col));

}

void SerialMatTranspose(Matrix& A) {
/*-----------------------------------------------------------
SerialMatTransposeKernel

    Description: Matrix transpose operation utilizing 
    sequential method on CPU.

    Parameters:

        A : Matrix&
            Matrix to transpose. Modifies A to contain data
            in transposed format.

-----------------------------------------------------------*/

    // Creating a copy to write into.

    // - In hindsight I should have just had the user pass in
    //   a buffer to keep things consistent with the kernels.

    Matrix A_T = A.copy();

    // Running over rows and cols
    for (int i = 0; i < A.height; i++) {
        for (int j = 0; j < A.width; j++) {
            // Doing the swap!
            A_T.data[i*A_T.stride + j] = A.data[j*A.stride + i];
        }
    }
    // Loading transposed data stored in A_T into A.
    A.load(A_T, MAT_CPU);
}


void MatTranspose(Matrix& A, int mode) {
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

    // Running on CPU. Serial modeeee.
    if (mode == 0) {

        cout << "// Starting serialMatTranspose..." << endl;

        // Starting clock...
        auto start = chrono::high_resolution_clock::now();

        SerialMatTranspose(A);

        // Stopping clock...
        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::nanoseconds>(stop - start);
        cout << "// End of serialMatTranspose..." << endl;

        cout.precision(6);
        cout << "// serialMatTranspose completed with an elapsed time of: ";

        // Converting from ns to ms. 
        cout << fixed << duration.count() * 0.000001 << " (ms)." << endl;
        cout.precision(0);
    }

    // Running Naively on GPU.
    else if (mode == 1) {

        // Allocate and load device Matrix d_A on GPU.
        Matrix d_A(A.height, A.width, A.stride, MAT_GPU);
        d_A.load(A, TO_DEV);

        // Allocate device Matrix d_AT onto GPU.
        Matrix d_AT(A.height, A.width, A.stride, MAT_GPU);

        dim3 blockDim(block_size, block_size);
        dim3 gridDim(max(A.width / blockDim.x, 1), max(A.height / blockDim.y, 1));

        cout << "// Launching 'Naive' MatTranspose Kernel..." << endl;
        cout << "// BlockDim (x,y): " << blockDim.x << " " << blockDim.y << endl;
        cout << "// GridDim (x,y): " << gridDim.x << " " << gridDim.y << endl;

        float duration;
        cudaEvent_t start, stop;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0);
        NaiveMatTransposeKernel<<<gridDim, blockDim>>>(d_A, d_AT);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&duration, start, stop);

        cudaDeviceSynchronize();
        cout << "// End of 'Naive' MatTranspose Kernel..." << endl;

        cout.precision(6);
        cout << "// 'Naive' MatTranspose completed with an elapsed time of ";
        cout << fixed << duration << " (ms)." << endl;
        cout.precision(0);

        // Loading transposed data back into A.
        A.load(d_AT, TO_HOST);

        // Cleaning up.
        d_A.dealloc();
        d_AT.dealloc();
    }

    // Running in shared-memory mode.
    else if (mode == 2) {

        // Allocate and load device Matrix d_A on GPU.
        Matrix d_A(A.height, A.width, A.stride, MAT_GPU);
        d_A.load(A, TO_DEV);

        // Allocate device Matrix d_AT onto GPU.
        Matrix d_AT(A.height, A.width, A.stride, MAT_GPU);

        dim3 blockDim(block_size, block_size);
        dim3 gridDim(max(A.width / blockDim.x, 1), max(A.height / blockDim.y, 1));

        cout << "// Launching 'Shared' MatTranspose Kernel..." << endl;
        cout << "// BlockDim (x,y): " << blockDim.x << " " << blockDim.y << endl;
        cout << "// GridDim (x,y): " << gridDim.x << " " << gridDim.y << endl;

        float duration;
        cudaEvent_t start, stop;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0);
        SharedMatTransposeKernel<<<gridDim, blockDim>>>(d_A, d_AT);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&duration, start, stop);

        cudaDeviceSynchronize();
        cout << "// End of 'Shared' MatTranspose Kernel..." << endl;

        cout.precision(6);
        cout << "// 'Shared' MatTranspose completed with an elapsed time of ";
        cout << fixed << duration << " (ms)." << endl;
        cout.precision(0);

        // Loading transposed data back into A.
        A.load(d_AT, TO_HOST);

        // Cleaning up.
        d_A.dealloc();
        d_AT.dealloc();
    }

    else {
        cout << "Unrecognized MatTranspose 'mode'..." << endl;
    }

}

