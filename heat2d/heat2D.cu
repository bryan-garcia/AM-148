#include "heat2D.h"


__global__
void initialize_grid_kernel(float* d_f_buff, float* d_x_buff, float* d_y_buff, int d_M, float d_xmin, float d_ymin, float delta) {

    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;

    float xt = d_xmin + ix * delta;
    float yt = d_ymin + iy * delta;

    if (ix >= d_M or iy >= d_M) return;

    d_x_buff[ix] = xt;
    d_y_buff[iy] = yt;
    d_f_buff[iy * d_M + ix] = expf(-0.5f * (xt * xt + yt * yt));

}

__global__
void FTCS_kernel(float* d_fval, const int M, const float dt, const float dxy, const float k) {

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col == 0 or col >= M) return;

    if (row == 0 or row >= M) return;

    // Operating only on inner cells.

    int ind = row * M + col;
    float coeff = k * dt / (dxy * dxy);
    float temp = d_fval[ind] + coeff * (d_fval[ind + 1] - 2 * d_fval[ind] + d_fval[ind - 1]) \
                             + coeff * (d_fval[ind + M] - 2 * d_fval[ind] + d_fval[ind - M]);

    d_fval[ind] = temp;

}

__global__
void BC_kernel(float* d_fval, int M) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (row >=M or col >= M) return;

    if (col == 0) {
        d_fval[row * M] = d_fval[row*M + 1]; 
    }
    if (col == M - 1) {
        d_fval[row * M + M - 1] = d_fval[row*M + M - 2];
    }
    if (row == 0) {
        d_fval[row * M + col] = d_fval[row * M + col + M];
    }
    if (row == M - 1) {
        d_fval[row * M + col] = d_fval[row * M + col - M];   
    }
}


void Heat2D::initialize_grid() {

    dim3 blockDim(TILE_W, TILE_H);
    dim3 gridDim((this->M + TILE_W - 1) / TILE_W, (this->M + TILE_W - 1) / TILE_H);

    cout << "Grid: " << gridDim.x << " " << gridDim.y << endl;
    initialize_grid_kernel<<<gridDim, blockDim>>>(this->d_fval, this->d_x, this->d_y, this->M, this->xlim.x, this->ylim.x, this->dxy);
    cudaDeviceSynchronize();

    this->update_x();
    this->update_y();
    this->update_fval();

}


void Heat2D::FTCS() {

    float duration;

    dim3 blockDim(TILE_W, TILE_H);
    dim3 gridDim((this->M + TILE_W - 1) / TILE_W, (this->M + TILE_W - 1) / TILE_H);

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    FTCS_kernel<<<gridDim, blockDim>>>(this->d_fval, this->M, this->dt, this->dxy, this->k);
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&duration, start, stop);

    cudaDeviceSynchronize();

    this->execution_time += duration;
}

void Heat2D::BC() {

    float duration;
    
    dim3 blockDim(TILE_W, TILE_H);
    dim3 gridDim((this->M + TILE_W - 1) / TILE_W, (this->M + TILE_W - 1) / TILE_H);

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    BC_kernel<<<gridDim, blockDim>>>(this->d_fval, this->M);
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&duration, start, stop);

    cudaDeviceSynchronize();

    this->execution_time += duration;
}



void Heat2D::update(float* new_buff, float* old_buff, size_t buff_size, int dir) {

    if (dir == FROM_DEV) {
        cudaMemcpy(new_buff, old_buff, buff_size, cudaMemcpyDeviceToHost);
    }
    else if (dir == TO_DEV) {
        cudaMemcpy(new_buff, old_buff, buff_size, cudaMemcpyHostToDevice);
    }
}

void Heat2D::update_fval() {
    update(fval, d_fval, fval_buffer_size, FROM_DEV);
}

void Heat2D::update_x() {
    update(x, d_x, x_buffer_size, FROM_DEV);
}

void Heat2D::update_y() {
    update(y, d_y, y_buffer_size, FROM_DEV);
}

void Heat2D::deallocate() {

    delete this->fval;
    delete this->x;
    delete this->y;

    cudaFree(this->d_fval);
    cudaFree(this->d_x);
    cudaFree(this->d_y);
}

void Heat2D::print_fval(ofstream& fout) {

    int i, j;
    for (i = 0; i < M; i++) {
        for (j = 0; j < M-1; j++) {
            fout << fval[i * M + j] << ", ";
        }
        fout << fval[i * M + j] << endl;
    }
}

void Heat2D::print_x(ofstream& fout) {

    int i;
    for (i = 0; i < M - 1; i++)
        fout << x[i] << ", ";

    fout << x[i] << endl;
}

void Heat2D::print_y(ofstream& fout) {

    int i;
    for (i = 0; i < M-1; i++)
        fout << y[i] << ", ";
    fout << y[i] << endl;

}

void Heat2D::save_fval(string fname) {

    ofstream data_stream;
    data_stream.open(fname);
    print_fval(data_stream);
    data_stream.close();
    
}


void Heat2D::save_x(string fname) {

    ofstream data_stream;
    data_stream.open(fname);
    print_x(data_stream);
    data_stream.close();

}


void Heat2D::save_y(string fname) {

    ofstream data_stream;
    data_stream.open(fname);
    print_y(data_stream);
    data_stream.close();

}
