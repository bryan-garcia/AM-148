/*------------------------------------------------------------------
psuedo_ND.cu

    Description: Class for emulating some Numpy array functionality.

------------------------------------------------------------------*/
#include "psuedo_ND.h"

__host__ __device__
void psuedo_ND::allocate() {
    if (this->mode == HOST or this->mode == DEVICE) {
        this->data = new float[this->len];
    }
    else if (this->mode == FROM_HOST_TO_DEVICE) {
        cudaMalloc((void**)&(data), this->buffer_size);
    }
}

__host__ __device__
void psuedo_ND::load(const psuedo_ND& nd_obj, int direction, bool verbose) {

    size_t buffer_size = this->buffer_size <= nd_obj.buffer_size \
                            ? this->buffer_size : nd_obj.buffer_size;

    if (direction == FROM_HOST_TO_DEVICE) {
        if (cudaSuccess == cudaMemcpy(this->data, nd_obj.data, buffer_size, cudaMemcpyHostToDevice) and verbose) {
            if (verbose) { printf("Successful FROM_HOST_TO_DEVICE cudaMemcpy.\n"); }
        }
        else {
            if (verbose) { printf("FAILED FROM_HOST_TO_DEVICE cudaMemcpy.\n"); }
        }
    }
    else if (direction == FROM_DEVICE_TO_HOST) {
        if (cudaSuccess == cudaMemcpy(this->data, nd_obj.data, buffer_size, cudaMemcpyDeviceToHost)) {
            if (verbose) { printf("Successful FROM_DEVICE_TO_HOST cudaMemcpy.\n"); }
        }
        else {
            if (verbose) { printf("FAILED FROM_DEVICE_TO_HOST cudaMemcpy.\n"); }
        }
    }
    else if (direction == HOST or direction == DEVICE) {
        memcpy(this->data, nd_obj.data, buffer_size);
        if (verbose) printf("Successful local memcpy.\n");
    }
}

__host__
void psuedo_ND::copy(const psuedo_ND& nd_obj, int direction, bool verbose) {
    this->width = nd_obj.width;
    this->height = nd_obj.height;
    this->len = this->width * this->height;
    this->buffer_size = nd_obj.buffer_size;
    this->ij = nd_obj.ij;
    this->stride = this->ij ? this->width : this->height;
    this->load(nd_obj, direction, verbose);
}

__host__ __device__
void psuedo_ND::release() { 
        
    if (this->mode == HOST or this->mode == DEVICE) {
        delete this->data; 
    }
    else if (this->mode == FROM_HOST_TO_DEVICE) {
        cudaFree(this->data);
    }
}

__host__ __device__
void d_print(const psuedo_ND& nd_obj) {

    int i = 0;
    int j = 0;

    printf("\n");

    for (i = 0; i < nd_obj.height; i++) {

        printf("[");

        for (j = 0; j < nd_obj.width - 1; j++) 
            printf("%f\t", nd_obj(i, j));

        printf("%f]\n",nd_obj(i,j));
    }

    printf("\n");

}

__host__ __device__
void psuedo_ND::flatten() {
    this->reshape(this->len, 1);
}

__host__ __device__
void psuedo_ND::reshape(int new_width, int new_height) {
    this->width = new_width;
    this->height = new_height;
    this->stride = this->ij ? this->width : this->height;
}

__host__ __device__
psuedo_ND zeros(int width, int height, int mode) {

    psuedo_ND PND(width, height, mode);

    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
            PND(i, j) = 0.0;

    return PND;
}

__device__
psuedo_ND d_zeros(int width, int height) {
    return zeros(width, height, DEVICE);
}

__host__ __device__
psuedo_ND zeros_like(psuedo_ND& zlike) {
    return zeros(zlike.width, zlike.height);
}

__host__ __device__
psuedo_ND linspace(float start, float end, int num, int mode) {

    psuedo_ND PND(num, 1, mode);

    float dx = (end - start) / (num - 1);
    
    for (int i = 0; i < num; i++) PND[i] = start + i * dx;

    PND.step_size = dx;

    return PND;
}

__device__
psuedo_ND d_linspace(float start, float end, int num) {
    return linspace(start, end, num, DEVICE);
}

__host__ __device__
psuedo_ND meshgrid(psuedo_ND& in_X, psuedo_ND& in_Y, int return_tag, int mode) {

    psuedo_ND PND(in_Y.width, in_X.width, mode);

    if (return_tag == X_MESH) {
        for (int i = 0; i < in_X.width; i++)
            for (int j = 0; j < in_Y.width; j++)
                PND(i, j) = in_X[i];
    }

    else if (return_tag == Y_MESH) {
        for (int i = 0; i < in_X.width; i++)
            for (int j = 0; j < in_Y.width; j++)
                PND(i, j) = in_Y[j];
    }

    return PND;
}

__device__
psuedo_ND d_meshgrid(psuedo_ND& in_X, psuedo_ND& in_Y, int return_tag) {
    return meshgrid(in_X, in_Y, return_tag, DEVICE);
}

__host__ __device__
psuedo_ND pnd_mat_mul(psuedo_ND A, psuedo_ND B) {

    psuedo_ND C = zeros(B.width, A.height);
    for (int i = 0; i < A.height; i++) {
        for (int j = 0; j < B.width; j++) {

            // Performing dot product
            for (int k = 0; k < A.width; k++) {
                C.data[C.stride * i + j] += A.data[A.stride * i + k] * B.data[B.stride * k + j];
            }
        }
    }

    return C;
}

/*---------------------------------------------------------------------/

                    OVERLOADED OPERATORS

/---------------------------------------------------------------------*/
__host__
std::ostream& operator<<(std::ostream& output, const psuedo_ND& nd_obj) {

    int i = 0;
    int j = 0;

    for (i = 0; i < nd_obj.height; i++) {

        output << "[";

        for (j = 0; j < nd_obj.width - 1; j++) 
            output << nd_obj(i, j) << ",\t";

        output << nd_obj(i,j) << "]" << std::endl;
    }

    output << std::endl;

    return output;
}

__host__ __device__
float& psuedo_ND::operator[](int index) const {

    if (abs(index) >= len) index = 0;

    else if (index < 0) index = this->len + index;

    return this->data[index];

}


__host__ __device__
float& psuedo_ND::operator()(int i, int j) const {

    int index;
    bool i_valid = abs(i) < this->height;
    bool j_valid = abs(j) < this->width;
    bool ij_valid = i_valid and j_valid;

    if (i >= 0 and j >= 0  and ij_valid) 
        index = this->stride * i + j;

    else if (i < 0 and i_valid)    
        index = ( (this->height - 1) + i) * stride + j;

    else if (j < 0 and j_valid)
        index = this->stride * i + ( (this->width - 1) + j);

    else if (i < 0 and j < 0 and ij_valid)
        index = ( (this->height - 1) + i) * stride + ( (this->width - 1) + j);

    else
        index = 0;

    return this->data[index];

}

__host__ __device__
void psuedo_ND::operator=(const psuedo_ND& nd_obj) {

    if (this != &nd_obj) {
        this->release();
        this->width = nd_obj.width;
        this->height = nd_obj.height;
        this->len = nd_obj.len;
        this->ij = nd_obj.ij;
        this->stride = this->ij ? this->width : this->height;
        this->allocate();
        this->load(nd_obj, this->mode);
    }
}


__host__ __device__
psuedo_ND psuedo_ND::operator+=(const psuedo_ND& nd_obj) {

    if (height == nd_obj.height and width == nd_obj.width) {

        for (int i = 0; i < height; i++) 
            for (int j = 0; j < width; j++)
                this->operator()(i,j) += nd_obj(i, j);

            return *this;
    }

}

__host__ __device__
psuedo_ND psuedo_ND::operator+=(float scalar) {

    for (int i = 0; i < height; i++) 
        for (int j = 0; j < width; j++)
            this->operator()(i,j) += scalar;

        return *this;

}

// __host__ __device__
// psuedo_ND psuedo_ND::operator+=(float scalar) {

//     for (int i = 0; i < height; i++) 
//         for (int j = 0; j < width; j++)
//             this->operator()(i,j) += scalar;

//         return *this;

// }

__host__ __device__
psuedo_ND psuedo_ND::operator+=(int scalar) {

    for (int i = 0; i < height; i++) 
        for (int j = 0; j < width; j++)
            this->operator()(i,j) += scalar;

        return *this;

}

__host__ __device__
psuedo_ND operator+(const psuedo_ND& nd_lhs, const psuedo_ND& nd_rhs) {

    if (nd_lhs.height == nd_rhs.height and nd_lhs.width == nd_rhs.width) {
        
        psuedo_ND PND(nd_lhs.width, nd_lhs.height);

        for (int i = 0; i < nd_lhs.height; i++) 
            for (int j = 0; j < nd_lhs.width; j++)
                PND(i,j) = nd_lhs(i, j) + nd_rhs(i, j);

        return PND;

    }
}

__host__ __device__
psuedo_ND operator+(const psuedo_ND& nd_lhs, float scalar) {
        
    psuedo_ND PND(nd_lhs.width, nd_lhs.height);

    for (int i = 0; i < nd_lhs.height; i++) 
        for (int j = 0; j < nd_lhs.width; j++)
            PND(i,j) = nd_lhs(i, j) + scalar;

    return PND;

}

// __host__ __device__
// psuedo_ND operator+(const psuedo_ND& nd_lhs, float scalar) {

//     psuedo_ND PND(nd_lhs.width, nd_lhs.height);

//     for (int i = 0; i < nd_lhs.height; i++) 
//         for (int j = 0; j < nd_lhs.width; j++)
//             PND(i,j) = nd_lhs(i, j) + scalar;

//     return PND;

// }

__host__ __device__
psuedo_ND operator+(const psuedo_ND& nd_lhs, int scalar) {
        
    psuedo_ND PND(nd_lhs.width, nd_lhs.height);

    for (int i = 0; i < nd_lhs.height; i++) 
        for (int j = 0; j < nd_lhs.width; j++)
            PND(i,j) = nd_lhs(i, j) + scalar;

    return PND;

}

__host__ __device__
psuedo_ND psuedo_ND::operator-=(const psuedo_ND& nd_obj) {

    if (height == nd_obj.height and width == nd_obj.width) {

        for (int i = 0; i < height; i++) 
            for (int j = 0; j < width; j++)
                this->operator()(i,j) -= nd_obj(i, j);

            return *this;
    }

}

__host__ __device__
psuedo_ND psuedo_ND::operator-=(float scalar) {

    for (int i = 0; i < height; i++) 
        for (int j = 0; j < width; j++)
            this->operator()(i,j) -= scalar;

        return *this;

}

// __host__ __device__
// psuedo_ND psuedo_ND::operator-=(float scalar) {

//     for (int i = 0; i < height; i++) 
//         for (int j = 0; j < width; j++)
//             this->operator()(i,j) -= scalar;

//         return *this;

// }

__host__ __device__
psuedo_ND psuedo_ND::operator-=(int scalar) {

    for (int i = 0; i < height; i++) 
        for (int j = 0; j < width; j++)
            this->operator()(i,j) -= scalar;

        return *this;

}

__host__ __device__
psuedo_ND operator-(const psuedo_ND& nd_lhs, const psuedo_ND& nd_rhs) {

    if (nd_lhs.height == nd_rhs.height and nd_lhs.width == nd_rhs.width) {
        
        psuedo_ND PND(nd_lhs.width, nd_lhs.height);

        for (int i = 0; i < nd_lhs.height; i++) 
            for (int j = 0; j < nd_lhs.width; j++)
                PND(i,j) = nd_lhs(i, j) - nd_rhs(i, j);

        return PND;

    }
}

__host__ __device__
psuedo_ND operator-(const psuedo_ND& nd_lhs, float scalar) {
        
    psuedo_ND PND(nd_lhs.width, nd_lhs.height);

    for (int i = 0; i < nd_lhs.height; i++) 
        for (int j = 0; j < nd_lhs.width; j++)
            PND(i,j) = nd_lhs(i, j) - scalar;

    return PND;

}

// __host__ __device__
// psuedo_ND operator-(const psuedo_ND& nd_lhs, float scalar) {

//     psuedo_ND PND(nd_lhs.width, nd_lhs.height);

//     for (int i = 0; i < nd_lhs.height; i++) 
//         for (int j = 0; j < nd_lhs.width; j++)
//             PND(i,j) = nd_lhs(i, j) - scalar;

//     return PND;

// }

__host__ __device__
psuedo_ND operator-(const psuedo_ND& nd_lhs, int scalar) {
        
    psuedo_ND PND(nd_lhs.width, nd_lhs.height);

    for (int i = 0; i < nd_lhs.height; i++) 
        for (int j = 0; j < nd_lhs.width; j++)
            PND(i,j) = nd_lhs(i, j) - scalar;

    return PND;

}

__host__ __device__
psuedo_ND psuedo_ND::operator*=(const psuedo_ND& nd_obj) {

    if (height == nd_obj.height and width == nd_obj.width) {

        for (int i = 0; i < height; i++) 
            for (int j = 0; j < width; j++)
                this->operator()(i,j) *= nd_obj(i, j);

            return *this;
    }

}

__host__ __device__
psuedo_ND psuedo_ND::operator*=(float scalar) {

    for (int i = 0; i < height; i++) 
        for (int j = 0; j < width; j++)
            this->operator()(i,j) *= scalar;

        return *this;

}

// __host__ __device__
// psuedo_ND psuedo_ND::operator*=(float scalar) {

//     for (int i = 0; i < height; i++) 
//         for (int j = 0; j < width; j++)
//             this->operator()(i,j) *= scalar;

//         return *this;

// }

__host__ __device__
psuedo_ND psuedo_ND::operator*=(int scalar) {

    for (int i = 0; i < height; i++) 
        for (int j = 0; j < width; j++)
            this->operator()(i,j) *= scalar;

        return *this;

}

__host__ __device__
psuedo_ND operator*(const psuedo_ND& nd_lhs, const psuedo_ND& nd_rhs) {

    if (nd_lhs.height == nd_rhs.height and nd_lhs.width == nd_rhs.width) {
        
        psuedo_ND PND(nd_lhs.width, nd_lhs.height);

        for (int i = 0; i < nd_lhs.height; i++) 
            for (int j = 0; j < nd_lhs.width; j++)
                PND(i,j) = nd_lhs(i, j) * nd_rhs(i, j);

        return PND;

    }
}

__host__ __device__
psuedo_ND operator*(const psuedo_ND& nd_lhs, float scalar) {
        
    psuedo_ND PND(nd_lhs.width, nd_lhs.height);

    for (int i = 0; i < nd_lhs.height; i++) 
        for (int j = 0; j < nd_lhs.width; j++)
            PND(i,j) = nd_lhs(i, j) * scalar;

    return PND;

}

// __host__ __device__
// psuedo_ND operator*(const psuedo_ND& nd_lhs, float scalar) {

//     psuedo_ND PND(nd_lhs.width, nd_lhs.height);

//     for (int i = 0; i < nd_lhs.height; i++) 
//         for (int j = 0; j < nd_lhs.width; j++)
//             PND(i,j) = nd_lhs(i, j) * scalar;

//     return PND;

// }

__host__ __device__
psuedo_ND operator*(const psuedo_ND& nd_lhs, int scalar) {
        
    psuedo_ND PND(nd_lhs.width, nd_lhs.height);

    for (int i = 0; i < nd_lhs.height; i++) 
        for (int j = 0; j < nd_lhs.width; j++)
            PND(i,j) = nd_lhs(i, j) * scalar;

    return PND;

}

__host__ __device__
psuedo_ND psuedo_ND::operator/=(float scalar) {

    if (scalar != 0.0f) {
        for (int i = 0; i < height; i++) 
            for (int j = 0; j < width; j++)
                this->operator()(i,j) /= scalar;

            return *this;
    }

}

// __host__ __device__
// psuedo_ND psuedo_ND::operator/=(float scalar) {

//     if (scalar != 0.0) {
//         for (int i = 0; i < height; i++) 
//             for (int j = 0; j < width; j++)
//                 this->operator()(i,j) /= scalar;

//             return *this;
//     }

// }

__host__ __device__
psuedo_ND psuedo_ND::operator/=(int scalar) {

    if (scalar != 0) {
        for (int i = 0; i < height; i++) 
            for (int j = 0; j < width; j++)
                this->operator()(i,j) /= scalar;

            return *this;
    }

}

__host__ __device__
psuedo_ND operator/(const psuedo_ND& nd_lhs, float scalar) {
        
    if (scalar != 0.0f) {
        psuedo_ND PND(nd_lhs.width, nd_lhs.height);

        for (int i = 0; i < nd_lhs.height; i++) 
            for (int j = 0; j < nd_lhs.width; j++)
                PND(i,j) = nd_lhs(i, j) / scalar;

        return PND;
    }

}

// __host__ __device__
// psuedo_ND operator/(const psuedo_ND& nd_lhs, float scalar) {
        
//     if (scalar != 0.0) {
//         psuedo_ND PND(nd_lhs.width, nd_lhs.height);

//         for (int i = 0; i < nd_lhs.height; i++) 
//             for (int j = 0; j < nd_lhs.width; j++)
//                 PND(i,j) = nd_lhs(i, j) / scalar;

//         return PND;
//     }

// }

__host__ __device__
psuedo_ND operator/(const psuedo_ND& nd_lhs, int scalar) {
        
    if (scalar != 0) {
        psuedo_ND PND(nd_lhs.width, nd_lhs.height);

        for (int i = 0; i < nd_lhs.height; i++) 
            for (int j = 0; j < nd_lhs.width; j++)
                PND(i,j) = nd_lhs(i, j) / scalar;

        return PND;
    }

}