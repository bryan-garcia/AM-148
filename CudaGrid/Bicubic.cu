#include "Bicubic.h"

__host__ __device__
void Bicubic::release() {
    lhs_nd.release();
    rhs_nd.release();
    center_nd.release();
    r_vector.release();
    z_vector.release();
}

__host__ __device__ 
void Bicubic::setup_ND_data() {

    if (this->mode == HOST or this->mode == DEVICE) {

        this->lhs_nd = zeros(4, 4);
        this->rhs_nd = zeros(4, 4);
        this->center_nd = zeros(4, 4);
        this->r_vector = zeros(4, 1);
        this->z_vector = zeros(1, 4);

    }

    else if (this->mode == FROM_HOST_TO_DEVICE) {

        printf("##### BICUBIC SETUP H2D\n");
        this->lhs_nd.copy(zeros(4, 4), FROM_HOST_TO_DEVICE, true);
        this->rhs_nd.copy(zeros(4, 4), FROM_HOST_TO_DEVICE, true);
        this->center_nd.copy(zeros(4, 4), FROM_HOST_TO_DEVICE, true);
        this->r_vector.copy(zeros(4, 1), FROM_HOST_TO_DEVICE, true);
        this->z_vector.copy(zeros(1, 4), FROM_HOST_TO_DEVICE, true);

    }

}

__host__ __device__
void Bicubic::setup_coefficient_matrices() {

    psuedo_ND temp_mat = zeros(4, 4);

    // LHS coefficient matrix for bicubic interpolation
    temp_mat(0, 0) = 1.0; 
    temp_mat(0, 1) = 0.0;
    temp_mat(0, 2) = 0.0;
    temp_mat(0, 3) = 0.0;

    temp_mat(1, 0) = 0.0;
    temp_mat(1, 1) = 0.0;
    temp_mat(1, 2) = 1.0;
    temp_mat(1, 3) = 0.0;

    temp_mat(2, 0) = -3.0;
    temp_mat(2, 1) = 3.0;
    temp_mat(2, 2) = -2.0;
    temp_mat(2, 3) = -1.0;

    temp_mat(3, 0) = 2.0;
    temp_mat(3, 1) = -2.0;
    temp_mat(3, 2) = 1.0;
    temp_mat(3, 3) = 1.0;

    // Copy to our bicubic class according to mode
    printf("lhs setup\n");
    this->lhs_nd.copy(temp_mat, this->mode, true);

    // RHS coefficient matrix for bicubic interpolation
    temp_mat(0, 0) = 1.0; 
    temp_mat(0, 1) = 0.0;
    temp_mat(0, 2) = -3.0;
    temp_mat(0, 3) = 2.0;

    temp_mat(1, 0) = 0.0; 
    temp_mat(1, 1) = 0.0;
    temp_mat(1, 2) = 3.0;
    temp_mat(1, 3) = -2.0;

    temp_mat(2, 0) = 0.0; 
    temp_mat(2, 1) = 1.0;
    temp_mat(2, 2) = -2.0;
    temp_mat(2, 3) = 1.0;

    temp_mat(3, 0) = 0.0; 
    temp_mat(3, 1) = 0.0;
    temp_mat(3, 2) = -1.0;
    temp_mat(3, 3) = 1.0;

    // Copy to our bicubic class according to mode
    printf("rhs setup\n");
    this->rhs_nd.copy(temp_mat, this->mode, true);

    temp_mat.release();


}

__host__ __device__
void Bicubic::set_interp_data(psuedo_ND& to_fill, psuedo_ND v, psuedo_ND vr, psuedo_ND vz, psuedo_ND vrz) {
    
    v.reshape(2, 2);
    vr.reshape(2, 2);
    vz.reshape(2, 2);
    vrz.reshape(2, 2);

    to_fill(0, 0) = v(0, 0);
    to_fill(0, 1) = v(1, 0);
    to_fill(1, 0) = v(0, 1);
    to_fill(1, 1) = v(1, 1);

    to_fill(2, 0) = vr(0, 0);
    to_fill(2, 1) = vr(1, 0);
    to_fill(3, 0) = vr(0, 1);
    to_fill(3, 1) = vr(1, 1);

    to_fill(0, 2) = vz(0, 0);
    to_fill(0, 3) = vz(1, 0);
    to_fill(1, 2) = vz(0, 1);
    to_fill(1, 3) = vz(1, 1);

    to_fill(2, 2) = vrz(0, 0);
    to_fill(2, 3) = vrz(1, 0);
    to_fill(3, 2) = vrz(0, 1);
    to_fill(3, 3) = vrz(1, 1);

}

__host__ __device__
float Bicubic::interp_point(float r, float z, psuedo_ND& fval_nd, psuedo_ND v_data, psuedo_ND vr_data, psuedo_ND vz_data, psuedo_ND vrz_data, int mode) {
    
    this->set_interp_data(fval_nd, v_data, vr_data, vz_data, vrz_data);

    // compute f(r, z)
    if (mode == 0) {
        this->r_vector[0] = 1;
        this->r_vector[1] = r;
        this->r_vector[2] = r * r;
        this->r_vector[3] = r * r * r;

        this->z_vector[0] = 1;
        this->z_vector[1] = z;
        this->z_vector[2] = z * z;
        this->z_vector[3] = z * z * z;
    }

    // compute f_{r}(r, z)
    else if (mode == 1) {
        this->r_vector[0] = 0;
        this->r_vector[1] = 1;
        this->r_vector[2] = 2 * r;
        this->r_vector[3] = 3 * r * r;

        this->z_vector[0] = 1;
        this->z_vector[1] = z;
        this->z_vector[2] = z * z;
        this->z_vector[3] = z * z * z;
    }

    // compute f_{z}(r, z)
    else if (mode == 2) {
        this->r_vector[0] = 1;
        this->r_vector[1] = r;
        this->r_vector[2] = r * r;
        this->r_vector[3] = r * r * r;

        this->z_vector[0] = 0;
        this->z_vector[1] = 1;
        this->z_vector[2] = 2 * z;
        this->z_vector[3] = 3 * z * z;
    }

    return this->compute_bicubic(fval_nd);

}

__host__ __device__
float Bicubic::compute_bicubic(psuedo_ND& fval_nd) {

    // 4 x 4 matrix of coefficients "a" corresponds to:
    //      A_coeff = pnd_mat_mul(this->lhs_nd, pnd_mat_mul(this->center_nd, this->rhs_nd));

    // 1 x 1 matrix containing interpolated p(r, z) polynomial value
    
    this->interpolant = pnd_mat_mul(this->r_vector, \
        pnd_mat_mul(pnd_mat_mul(this->lhs_nd, pnd_mat_mul(fval_nd, this->rhs_nd)), \
            this->z_vector))[0];

    return this->interpolant;
    
}
