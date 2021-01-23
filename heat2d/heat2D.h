#ifndef _HEAT_2D_H_
#define _HEAT_2D_H_

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
using namespace std;

#define FROM_DEV 0
#define TO_DEV 1

#define TILE_W 16
#define TILE_H 16
#define R 1
#define BLOCK_W (TILE_W + (2 * R))
#define BLOCK_H (TILE_H + (2 * R))


class Heat2D {

    public:

        float2 xlim;
        float2 ylim;
        int M;

        float dxy;
        float dt;
        float t;
        float tmax;
        float k;

        float* fval;
        float* x;
        float* y;

        float* d_fval;
        float* d_x;
        float* d_y;


        float execution_time;

        size_t fval_buffer_size;
        size_t x_buffer_size;
        size_t y_buffer_size;

        Heat2D(float2 xlim, float2 ylim, const int M, const float tmax) {

            this->xlim = xlim;
            this->ylim = ylim;
            this->tmax = tmax;
            this->M = M;
            this->execution_time = 0.0f;
            
            fval_buffer_size = M * M * sizeof(float);
            x_buffer_size = M * sizeof(float);
            y_buffer_size = M * sizeof(float);

            fval = (float*) malloc(fval_buffer_size);
            x = (float*) malloc(x_buffer_size);
            y = (float*) malloc(y_buffer_size);

            cudaMalloc((void**) &d_fval, fval_buffer_size);
            cudaMalloc((void**) &d_x, x_buffer_size);
            cudaMalloc((void**) &d_y, y_buffer_size);

            // Heat coefficient
            k = 1.f;
            
            // Integrator constants
            dxy = (xlim.y - xlim.x) / float(M);
            dt = dxy * dxy / (8.f * k);
        }

        void update(float* new_buff, float* old_buff, size_t buff_size, int dir);
        void update_fval();
        void update_x();
        void update_y();

        void print_x(ofstream& fout);
        void print_y(ofstream& fout);
        void print_fval(ofstream& fout);

        void save_fval(string fname);
        void save_x(string fname);
        void save_y(string fname);

        void deallocate();

        void initialize_grid();
        
        void FTCS();
        void BC();

};


#endif
