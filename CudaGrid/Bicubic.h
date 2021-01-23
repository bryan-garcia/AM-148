#ifndef BICUBIC_H
#define BICUBIC_H

#include "psuedo_ND.h"

class Bicubic {

    public:

        int mode;
        float interpolant;

        psuedo_ND lhs_nd;
        psuedo_ND rhs_nd;
        psuedo_ND center_nd;
        psuedo_ND r_vector;
        psuedo_ND z_vector;

        __host__ __device__
        Bicubic(int mode = HOST) 
            : lhs_nd(4, 4, mode), rhs_nd(4, 4, mode), center_nd(4, 4, mode), 
              r_vector(4, 1, mode), z_vector(1, 4, mode) {

            this->mode = mode;
            setup_ND_data();
            setup_coefficient_matrices();

        }

        __host__ __device__
        void setup_ND_data();

        __host__ __device__
        void setup_coefficient_matrices();

        __host__ __device__
        void set_interp_data(psuedo_ND& fval_nd, psuedo_ND v, psuedo_ND vr, psuedo_ND vz, psuedo_ND vrz);

        __host__ __device__
        void release();

        __host__ __device__
        float interp_point(float r, float z, psuedo_ND& fval_nd,\
            psuedo_ND v_data, psuedo_ND vr_data, psuedo_ND vz_data, psuedo_ND vrz_data, int mode = 0);

        __host__ __device__
        float compute_bicubic(psuedo_ND& fval_nd);

};

#endif