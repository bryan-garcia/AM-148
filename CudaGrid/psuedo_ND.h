#ifndef _PSUEDO_ND_H_
#define _PSUEDO_ND_H_

#define X_MESH 0
#define Y_MESH 1
#define R_MESH X_MESH
#define Z_MESH Y_MESH
#define HOST 0
#define DEVICE 1
#define FROM_DEVICE_TO_HOST 2
#define FROM_HOST_TO_DEVICE 3 

#define TRUE 1
#define FALSE 0
#define VERBOSE_FLAG FALSE

#include <iostream>
#include <cstdio>
#include <cmath>


class psuedo_ND {
    public:

        int height;
        int width;
        int len;
        int buffer_size;
        int stride;
        bool ij;
        float step_size;
        float* data;

        int mode;

        __host__ __device__
        psuedo_ND(int width, int height, int mode = HOST, bool ij = true) {
            this->width = width;
            this->height = height;
            this->len = width * height;
            this->buffer_size = this->len * sizeof(float);
            this->ij = ij;
            this->mode = mode;
            this->stride = ij ? width : height;
            this->allocate();
        };

        __host__ __device__ void flatten();
        __host__ __device__ void reshape(int new_width, int new_height);
        __host__ __device__ void allocate();
        __host__ __device__ void load(const psuedo_ND& nd_obj, int direction, bool verbose = VERBOSE_FLAG);
        __host__ __device__ void release();
        __host__ __device__ void copy(const psuedo_ND& nd_obj, int direction, bool verbose = VERBOSE_FLAG);

        __host__
        friend std::ostream& operator<<(std::ostream& output, const psuedo_ND& nd_obj);

        __host__ __device__
        float& operator[](int index) const;
        __host__ __device__
        float& operator()(int i, int j) const;

        __host__ __device__
        void operator=(const psuedo_ND& nd_obj);

        __host__ __device__
        psuedo_ND operator+=(const psuedo_ND& nd_obj);
        __host__ __device__
        psuedo_ND operator+=(float scalar);
        // __host__ __device__
        // psuedo_ND operator+=(float scalar);
        __host__ __device__
        psuedo_ND operator+=(int scalar);
        __host__ __device__
        friend psuedo_ND operator+(const psuedo_ND& lhs, const psuedo_ND& rhs);
        __host__ __device__
        friend psuedo_ND operator+(const psuedo_ND& lhs, float scalar);
        // __host__ __device__
        // friend psuedo_ND operator+(const psuedo_ND& lhs, float scalar);
        __host__ __device__
        friend psuedo_ND operator+(const psuedo_ND& lhs, int scalar);
        
        __host__ __device__
        psuedo_ND operator-=(const psuedo_ND& nd_obj);
        __host__ __device__
        psuedo_ND operator-=(float scalar);
        // __host__ __device__
        // psuedo_ND operator-=(float scalar);
        __host__ __device__
        psuedo_ND operator-=(int scalar);
        __host__ __device__
        friend psuedo_ND operator-(const psuedo_ND& lhs, const psuedo_ND& rhs);
        __host__ __device__
        friend psuedo_ND operator-(const psuedo_ND& lhs, float scalar);
        // __host__ __device__
        // friend psuedo_ND operator-(const psuedo_ND& lhs, float scalar);
        __host__ __device__
        friend psuedo_ND operator-(const psuedo_ND& lhs, int scalar);

        __host__ __device__
        psuedo_ND operator*=(const psuedo_ND& nd_obj);
        __host__ __device__
        psuedo_ND operator*=(float scalar);
        // __host__ __device__
        // psuedo_ND operator*=(float scalar);
        __host__ __device__
        psuedo_ND operator*=(int scalar);
        __host__ __device__
        friend psuedo_ND operator*(const psuedo_ND& lhs, const psuedo_ND& rhs);
        __host__ __device__
        friend psuedo_ND operator*(const psuedo_ND& lhs, float scalar);
        // __host__ __device__
        // friend psuedo_ND operator*(const psuedo_ND& lhs, float scalar);
        __host__ __device__
        friend psuedo_ND operator*(const psuedo_ND& lhs, int scalar);

        __host__ __device__
        psuedo_ND operator/=(float scalar);
        // __host__ __device__
        // psuedo_ND operator/=(float scalar);
        __host__ __device__
        psuedo_ND operator/=(int scalar);
        __host__ __device__
        friend psuedo_ND operator/(const psuedo_ND& lhs, float scalar);
        // __host__ __device__
        // friend psuedo_ND operator/(const psuedo_ND& lhs, float scalar);
        __host__ __device__
        friend psuedo_ND operator/(const psuedo_ND& lhs, int scalar);

};

__host__ __device__
psuedo_ND pnd_mat_mul(psuedo_ND A, psuedo_ND B);

__host__ __device__
psuedo_ND zeros(int width, int height, int mode = HOST);

__device__
psuedo_ND d_zeros(int width, int height);

__host__ __device__
psuedo_ND zeros_like(psuedo_ND& zlike);

__host__ __device__
psuedo_ND linspace(float start, float stop, int num, int mode = HOST);

__device__
psuedo_ND d_linspace(float start, float stop, int num);

__host__ __device__
psuedo_ND meshgrid(psuedo_ND& in_X, psuedo_ND& in_Y, int return_tag, int mode = HOST);

__device__
psuedo_ND d_meshgrid(psuedo_ND& in_X, psuedo_ND& in_Y, int return_tag);

__host__ __device__
void d_print(const psuedo_ND& nd_obj);


#endif