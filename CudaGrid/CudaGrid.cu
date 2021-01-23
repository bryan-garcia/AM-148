#include "CudaGrid.h"

__host__ __device__
void CudaGrid::release() {

    r.release();
    z.release();
    v.release();
    vr.release();
    vz.release();
    vrz.release();
    r_nearest_cell.release();
    z_nearest_cell.release();
    v_cell.release();
    vr_cell.release();
    vz_cell.release();
    vrz_cell.release();
    fval_nd.release();
    lhs_nd.release();
    rhs_nd.release();
    center_nd.release();
    r_vector.release();
    z_vector.release();

}

__device__
void CudaGrid::setup_interp_data_gpu(psuedo_ND to_fill, psuedo_ND v, psuedo_ND vr, psuedo_ND vz, psuedo_ND vrz) {
    
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
void CudaGrid::setup_coefficient_matrices() {

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
    this->lhs_nd.copy(temp_mat, this->mode);

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
    this->rhs_nd.copy(temp_mat, this->mode);

    temp_mat.release();


}

__host__ __device__
void CudaGrid::setup_grid_data() {

    if (this->mode == HOST or this->mode == DEVICE) {
        psuedo_ND _r = linspace(this->rmin, this->rmax, this->nr);
        psuedo_ND _z = linspace(this->zmin, this->zmax, this->nz);
        
        this->dr = abs(_r[1] - _r[0]);
        this->dz = abs(_z[1] - _z[0]);

        this->r = meshgrid(_r, _z, R_MESH);
        this->z = meshgrid(_r, _z, Z_MESH);

        _r.release(); _z.release();

        this->v = zeros(this->nr, this->nz);
        this->vr = zeros(this->nr, this->nz);
        this->vz = zeros(this->nr, this->nz);
        this->vrz = zeros(this->nr, this->nz);

        this->r_nearest_cell = zeros(4, 1);
        this->z_nearest_cell = zeros(4, 1);

        this->v_cell = zeros(4, 1);
        this->vr_cell = zeros(4, 1);
        this->vz_cell = zeros(4, 1); 
        this->vrz_cell = zeros(4, 1); 

        this->fval_nd = zeros(4, 4);


        // Bicubic stuff
        this->center_nd = zeros(4, 4);
        this->r_vector = zeros(4, 1);
        this->z_vector = zeros(1, 4);

    }

    else if (this->mode == FROM_HOST_TO_DEVICE) {
        psuedo_ND _r = linspace(this->rmin, this->rmax, this->nr);
        psuedo_ND _z = linspace(this->zmin, this->zmax, this->nz);
        
        this->dr = abs(_r[1] - _r[0]);
        this->dz = abs(_z[1] - _z[0]);

        this->r.copy(meshgrid(_r, _z, R_MESH), FROM_HOST_TO_DEVICE);
        this->z.copy(meshgrid(_r, _z, Z_MESH), FROM_HOST_TO_DEVICE);
        
        _r.release(); _z.release();

        this->v.copy(zeros(this->nr, this->nz), FROM_HOST_TO_DEVICE);
        this->vr.copy(zeros(this->nr, this->nz), FROM_HOST_TO_DEVICE);
        this->vz.copy(zeros(this->nr, this->nz), FROM_HOST_TO_DEVICE);
        this->vrz.copy(zeros(this->nr, this->nz), FROM_HOST_TO_DEVICE);

        this->r_nearest_cell.copy(zeros(4, 1), FROM_HOST_TO_DEVICE);
        this->z_nearest_cell.copy(zeros(4, 1), FROM_HOST_TO_DEVICE);

        this->v_cell.copy(zeros(4, 1), FROM_HOST_TO_DEVICE);
        this->vr_cell.copy(zeros(4, 1), FROM_HOST_TO_DEVICE);
        this->vz_cell.copy(zeros(4, 1), FROM_HOST_TO_DEVICE);
        this->vrz_cell.copy(zeros(4, 1), FROM_HOST_TO_DEVICE);

        this->fval_nd.copy(zeros(4, 4), FROM_HOST_TO_DEVICE);

        // Bicubic stuff
        this->lhs_nd.copy(zeros(4, 4), FROM_HOST_TO_DEVICE);
        this->rhs_nd.copy(zeros(4, 4), FROM_HOST_TO_DEVICE);
        this->center_nd.copy(zeros(4, 4), FROM_HOST_TO_DEVICE);
        this->r_vector.copy(zeros(4, 1), FROM_HOST_TO_DEVICE);
        this->z_vector.copy(zeros(1, 4), FROM_HOST_TO_DEVICE);
    }

    this->setup_coefficient_matrices();
}

__host__ __device__
void CudaGrid::copy(const CudaGrid& efit_obj, int mode, bool verbose) {

    this->nr = efit_obj.nr;
    this->nz = efit_obj.nz;
    this->rmin = efit_obj.rmin;
    this->rmax = efit_obj.rmax;
    this->zmin = efit_obj.zmin;
    this->zmax = efit_obj.zmax;
    this->dr = efit_obj.dr;
    this->dz = efit_obj.dz;
    this->load(efit_obj, mode, verbose);
}


__host__
void CudaGrid::load(const CudaGrid& efit_obj, int direction, bool verbose) {
    
    if (verbose) printf("\n# Loading this->r.data with efit_obj.r.data... ");
    this->r.load(efit_obj.r, direction, verbose);

    if (verbose) printf("\n# Loading this->z.data with efit_obj.z.data... ");
    this->z.load(efit_obj.z, direction, verbose);

    if (verbose) printf("\n# Loading this->v.data with efit_obj.v.data... ");
    this->v.load(efit_obj.v, direction, verbose);

    if (verbose) printf("\n# Loading this->vr.data with efit_obj.vr.data... ");
    this->vr.load(efit_obj.vr, direction, verbose);

    if (verbose) printf("\n# Loading this->vz.data with efit_obj.vz.data... ");
    this->vz.load(efit_obj.vz, direction, verbose);

    if (verbose) printf("\n# Loading this->vrz.data with efit_obj.vrz.data...\n");
    this->vrz.load(efit_obj.vrz, direction, verbose);

    if (verbose) printf("\n# Loading this->r_nearest_cell.data with efit_obj.r_nearest_cell.data...\n");
    this->r_nearest_cell.load(efit_obj.r_nearest_cell, direction, verbose);

    if (verbose) printf("\n# Loading this->z_nearest_cell.data with efit_obj.z_nearest_cell.data...\n");
    this->z_nearest_cell.load(efit_obj.z_nearest_cell, direction, verbose);

    if (verbose) printf("\n# Loading this->v_cell.data with efit_obj.v_cell.data...\n");
    this->v_cell.load(efit_obj.v_cell, direction, verbose);

    if (verbose) printf("\n# Loading this->vr_cell.data with efit_obj.vr_cell.data...\n");
    this->vr_cell.load(efit_obj.vr_cell, direction, verbose);

    if (verbose) printf("\n# Loading this->vz_cell.data with efit_obj.vz_cell.data...\n");
    this->vz_cell.load(efit_obj.vz_cell, direction, verbose);

    if (verbose) printf("\n# Loading this->vrz_cell.data with efit_obj.vrz_cell.data...\n");
    this->vrz_cell.load(efit_obj.vrz_cell, direction, verbose);

    if (verbose) printf("\n# Loading this->fval_nd.data with efit_obj.fval_nd.data...\n");
    this->fval_nd.load(efit_obj.fval_nd, direction, verbose);

    if (verbose) printf("\n# Loading this->lhs_nd.data with efit_obj.lhs_nd.data...\n");
    this->lhs_nd.load(efit_obj.lhs_nd, direction, verbose);

    if (verbose) printf("\n# Loading this->rhs_nd.data with efit_obj.rhs_nd.data...\n");
    this->rhs_nd.load(efit_obj.rhs_nd, direction, verbose);

    if (verbose) printf("\n# Loading this->center_nd.data with efit_obj.center_nd.data...\n");
    this->center_nd.load(efit_obj.center_nd, direction, verbose);

    if (verbose) printf("\n# Loading this->r_vector.data with efit_obj.r_vector.data...\n");
    this->r_vector.load(efit_obj.r_vector, direction, verbose);

    if (verbose) printf("\n# Loading this->z_vector.data with efit_obj.z_vector.data...\n");
    this->z_vector.load(efit_obj.z_vector, direction, verbose);

}



__host__ __device__
void CudaGrid::find_nearest_cell(float rval, float zval) {
    
    int r_ind;
    int z_ind;

    r_ind = (int) max(min(int(floor((rval - this->rmin) / this->dr)), this->nr - 2), 0);
    z_ind = (int) max(min(int(floor((zval - this->zmin) / this->dz)), this->nz - 2), 0);

    this->r_nearest_cell[0] = r_ind;
    this->r_nearest_cell[1] = r_ind + 1;
    this->r_nearest_cell[2] = r_ind;
    this->r_nearest_cell[3] = r_ind + 1;

    this->z_nearest_cell[0] = z_ind;
    this->z_nearest_cell[1] = z_ind;
    this->z_nearest_cell[2] = z_ind + 1;
    this->z_nearest_cell[3] = z_ind + 1;

}

__device__
void CudaGrid::find_nearest_cell_gpu(float rval, float zval, psuedo_ND& temp_r, psuedo_ND& temp_z) {
    
    int r_ind;
    int z_ind;

    r_ind = (int) fmax(fmin( (float) floorf((rval - this->rmin) / this->dr), (float) this->nr - 2), (float) 0.0);
    z_ind = (int) fmax(fmin( (float) floorf((zval - this->zmin) / this->dz), (float) this->nz - 2), (float) 0.0);

    temp_r[0] = r_ind;
    temp_r[1] = r_ind + 1;
    temp_r[2] = r_ind;
    temp_r[3] = r_ind + 1;

    temp_z[0] = z_ind;
    temp_z[1] = z_ind;
    temp_z[2] = z_ind + 1;
    temp_z[3] = z_ind + 1;

    printf("---- IN FNC: %d\n", z_ind);


}

__host__ 
void init_data(CudaGrid& host_grid, int option) {

    CudaGrid d_grid(host_grid.rmin, host_grid.rmax,
                host_grid.zmin, host_grid.zmax,
                host_grid.nr, host_grid.nz, FROM_HOST_TO_DEVICE);

    d_grid.copy(host_grid, FROM_HOST_TO_DEVICE);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(host_grid.nr / blockDim.x + 1, host_grid.nz / blockDim.y + 1);

    init_data_kernel<<<gridDim, blockDim>>>(d_grid, option);

    cudaDeviceSynchronize();

    host_grid.copy(d_grid, FROM_DEVICE_TO_HOST);

    d_grid.release();

}

__global__
void init_data_kernel(CudaGrid in_grid, int option) {

    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;

    if (i < in_grid.nr and j < in_grid.nz) {

        if (option == 1) 
            in_grid.v(i,j) = test_f1(in_grid.r(i, j), in_grid.z(i, j));
        else if (option == 2)
            in_grid.v(i,j) = test_f2(in_grid.r(i, j), in_grid.z(i, j)); 
        else if (option == 3)
            in_grid.v(i,j) = test_f3(in_grid.r(i, j), in_grid.z(i, j));
    }

}


__host__ 
void compute_pderivs(CudaGrid& host_grid, bool unit_spacing) {

    float duration;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    CudaGrid d_grid(host_grid.rmin, host_grid.rmax,
                host_grid.zmin, host_grid.zmax,
                host_grid.nr, host_grid.nz, FROM_HOST_TO_DEVICE);

    d_grid.copy(host_grid, FROM_HOST_TO_DEVICE);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(host_grid.nr / blockDim.x + 1, host_grid.nz / blockDim.y + 1);

    cudaEventRecord(start, 0);
    kernel_compute_pderivs_all<<<gridDim, blockDim>>>(d_grid, unit_spacing);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&duration, start, stop);
    cudaDeviceSynchronize();

    host_grid.copy(d_grid, FROM_DEVICE_TO_HOST);

    std::cout.precision(6);
    std::cout << "//compute_pderivs completed with an elapsed time of ";
    std::cout << std::fixed << duration << " (ms)." << std::endl;

    d_grid.release();

}

__global__
void kernel_compute_pderivs_all(CudaGrid in_grid, bool unit_spacing) {

    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;

    int nr = in_grid.nr;
    int nz = in_grid.nz;

    // Compute pderivs within bounds
    if (i < nr and j < nz) {

        float dr = unit_spacing ? 1.0 : in_grid.dr;
        float dz = unit_spacing ? 1.0 : in_grid.dz;

        // Can perform central difference formula

        // Check if we are interior points of the domain
        if ((i > 0 and i < nr - 1) and (j > 0 and j < nz - 1)) {
            in_grid.vr(i, j) = (in_grid.v(i + 1, j) - in_grid.v(i - 1, j)) / (2 * dr);
            in_grid.vz(i, j) = (in_grid.v(i, j + 1) - in_grid.v(i, j - 1)) / (2 * dz);
            in_grid.vrz(i, j) = (in_grid.v(i + 1, j + 1) - in_grid.v(i + 1, j - 1) \
                                - in_grid.v(i - 1, j + 1) + in_grid.v(i - 1, j - 1)) / (4 * dr * dz);
        }

        // If first or last row, compute partial_z along columns (missed on first pass)
        if (i == 0 or i == nr - 1) { 
            in_grid.vz(i, j) = (in_grid.v(i, j + 1) - in_grid.v(i, j - 1)) / (2 * dz);
        }
        
        // If first or last col, compute partial_r along row (missed on first pass) 
        if (j == 0 or j == nz - 1) {
            in_grid.vr(i, j) = (in_grid.v(i + 1, j) - in_grid.v(i - 1, j)) / (2 * dr);
        }

        // Forward difference scheme to compute partial_r along first row
        if (i == 0) {
            in_grid.vr(i, j) = (-3 * in_grid.v(i, j) 
                                + 4 * in_grid.v(i + 1, j) 
                                - in_grid.v(i + 2, j) ) / (2 * dr);

            
            // Compute mixed partials along first row as well
            if (j <= (nz - 1) - 2) {
                
                // Forward difference for r applied to forward difference of z
                in_grid.vrz(i, j) = (-3 * (-3*in_grid.v(i, j) + 4*in_grid.v(i, j + 1) - in_grid.v(i, j+2)) / (2*dz) \
                    + 4 * (-3*in_grid.v(i+1, j) + 4*in_grid.v(i+1, j + 1) - in_grid.v(i+1, j+2)) / (2 * dz) \
                    - (-3*in_grid.v(i+2, j) + 4*in_grid.v(i+2, j + 1) - in_grid.v(i+2, j+2)) / (2 * dz)) / (2 * dr);

            }

            else {

                // Forward difference for r applied to backward difference of z
                in_grid.vrz(i, j) = (-3 * (3*in_grid.v(i, j) - 4*in_grid.v(i, j - 1) + in_grid.v(i, j-2)) / (2*dz) \
                    + 4 * (3*in_grid.v(i+1, j) - 4*in_grid.v(i+1, j - 1) + in_grid.v(i+1, j-2)) / (2 * dz) \
                    - (3*in_grid.v(i+2, j) - 4*in_grid.v(i+2, j - 1) + in_grid.v(i+2, j-2)) / (2 * dz)) / (2 * dr);
            }
        }

        // Backward difference scheme to compute partial_r along last row
        if (i == nr - 1) {
            in_grid.vr(i, j) = (3 * in_grid.v(i, j) \
                                    -4 * in_grid.v(i - 1, j) \
                                    + in_grid.v(i - 2, j)) / (2 * dr);

            // Compute mixed partials along last row as well.
            if (j <= (nz - 1) - 2) {
                // Backward difference for i-axis applied to forward difference of j-axis
                in_grid.vrz(i, j) = (3 * (-3*in_grid.v(i, j) + 4*in_grid.v(i, j + 1) - in_grid.v(i, j+2)) / (2*dz) \
                    - 4 * (-3*in_grid.v(i-1, j) + 4*in_grid.v(i-1, j + 1) - in_grid.v(i-1, j+2)) / (2 * dz) \
                    + (-3*in_grid.v(i-2, j) + 4*in_grid.v(i-2, j + 1) - in_grid.v(i-2, j+2)) / (2 * dz)) / (2 * dr);
            }

            else {
                // Backward difference for i-axis applied to backward difference of j-axis
                in_grid.vrz(i, j) = (3 * (3*in_grid.v(i, j) - 4*in_grid.v(i, j - 1) + in_grid.v(i, j-2)) / (2*dz) \
                    - 4 * (3*in_grid.v(i-1, j) - 4*in_grid.v(i-1, j - 1) + in_grid.v(i-1, j-2)) / (2 * dz) \
                    + (3*in_grid.v(i-2, j) - 4*in_grid.v(i-2, j - 1) + in_grid.v(i-2, j-2)) / (2 * dz)) / (2 * dr);
            }

        }

        // Forward difference scheme to compute partial_z along first col
        if (j == 0) {
            in_grid.vz(i, j) = (-3 * in_grid.v(i, j) 
                                + 4 * in_grid.v(i, j + 1) 
                                - in_grid.v(i, j + 2) ) / (2 * dz);
            
            // Compute mixed partial along first column as well
            if (i > 0 and i < nr - 1) {

                // Apply centered difference along i-axis with j-axis forward difference data
                in_grid.vrz(i, j) = ( \
                    ((-3 * in_grid.v(i + 1, j) + 4 * in_grid.v(i + 1, j + 1) - in_grid.v(i + 1, j + 2) ) / (2 * dz)) \
                    - ((-3 * in_grid.v(i - 1, j) + 4 * in_grid.v(i - 1, j + 1) - in_grid.v(i - 1, j + 2) ) / (2 * dz)) \
                    ) / (2 * dr);
            }
        }

        // Backward difference scheme to compute partial_z along last col
        if (j == nz - 1) {
            in_grid.vz(i, j) = (3 * in_grid.v(i, j) \
                                    -4 * in_grid.v(i, j - 1) \
                                    + in_grid.v(i, j - 2)) / (2 * dz);
            
            // Compute mixed partial along last column as well
            if (i > 0 and i < nr - 1) {

                // Apply centered difference along i-axis with j-axis backward difference data
                in_grid.vrz(i, j) = ( \
                    ((3 * in_grid.v(i + 1, j) -4 * in_grid.v(i + 1, j - 1) + in_grid.v(i + 1, j - 2)) / (2 * dz)) \
                    - ((3 * in_grid.v(i - 1, j) -4 * in_grid.v(i - 1, j - 1) + in_grid.v(i - 1, j - 2)) / (2 * dz)) \
                    ) / (2 * dr);
            }
        }

    }

}


__host__
void bicubic_test(CudaGrid& crude_grid, CudaGrid& fine_grid, int option, bool unit_spacing) {
    
    float duration_single; 
    float duration_total = 0;

    CudaGrid d_crude(crude_grid.rmin, crude_grid.rmax,
                crude_grid.zmin, crude_grid.zmax,
                crude_grid.nr, crude_grid.nz, FROM_HOST_TO_DEVICE);

    CudaGrid d_fine(fine_grid.rmin, fine_grid.rmax,
                fine_grid.zmin, fine_grid.zmax,
                fine_grid.nr, fine_grid.nz, FROM_HOST_TO_DEVICE);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(fine_grid.nr / blockDim.x + 1, fine_grid.nz / blockDim.y + 1);
    init_data_kernel<<<gridDim, blockDim>>>(d_crude, option);
    kernel_compute_pderivs_all<<<gridDim, blockDim>>>(d_crude, unit_spacing);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < fine_grid.nr; i++) {
        for (int j = 0; j < fine_grid.nz; j++) {
            cudaEventRecord(start, 0);
            kernel_bicubic_init<<<gridDim, blockDim>>>(i, j, d_crude, d_fine, V);
            cudaEventRecord(stop, 0);
            cudaEventElapsedTime(&duration_single, start, stop);
            duration_total += duration_single;
        }
    }
    cudaDeviceSynchronize();

    fine_grid.copy(d_fine, FROM_DEVICE_TO_HOST);
    crude_grid.copy(d_crude, FROM_DEVICE_TO_HOST);

    d_fine.release();
    d_crude.release();

    std::cout.precision(6);
    std::cout << "-- BICUBIC INTERPOLATION OF " << fine_grid.nr * fine_grid.nz << " entries completed with an elapsed time of ";
    std::cout << std::fixed << duration_total << " (ms)." << std::endl;

}

__global__
void kernel_bicubic_init(int index_i, int index_j, CudaGrid crude, CudaGrid fine, int data_tag) {

    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;

    if (i == index_i and j == index_j) {

        float rval = fine.r.data[fine.r.stride * i + j];
        float zval = fine.z.data[fine.z.stride * i + j];

        psuedo_ND _r = zeros(4, 1);
        psuedo_ND _z = zeros(1, 4);
        psuedo_ND _r_cell = zeros(4, 1);
        psuedo_ND _z_cell = zeros(4, 1);
        psuedo_ND _fval = zeros(4, 4);
        psuedo_ND _v = zeros(4, 1);
        psuedo_ND _vr = zeros(4, 1);
        psuedo_ND _vz = zeros(4, 1);
        psuedo_ND _vrz = zeros(4, 1);


        int r_ind;
        int z_ind;

        // Computing the nearest crude grid r-index and z-index relative to rval and zval

        // max min makes sure we're within the crude grid domain of indices
        r_ind = (int) max(min( (float) floor((rval - crude.rmin) / crude.dr), (float) crude.nr - 2), (float) 0.0);
        z_ind = (int) max(min( (float) floor((zval - crude.zmin) / crude.dz), (float) crude.nz - 2), (float) 0.0);

        // Storing the nearest cell indices 
        // c - d
        //  \
        //   \
        // a - b
        // Ordered corners a -> b -> c -> d
        _r_cell.data[0] = (float) r_ind;
        _r_cell.data[1] = (float) r_ind + 1;
        _r_cell.data[2] = (float) r_ind;
        _r_cell.data[3] = (float) r_ind + 1;

        _z_cell.data[0] = (float) z_ind;
        _z_cell.data[1] = (float) z_ind;
        _z_cell.data[2] = (float) z_ind + 1;
        _z_cell.data[3] = (float) z_ind + 1;

        // Populate entries with function and partial derivative data surrounding the indices computed
        for (int k = 0; k < 4; k++) {
            _v[k] = crude.v.data[int(crude.v.stride * _r_cell[k] + _z_cell[k])];
            _vr[k] = crude.vr.data[int(crude.vr.stride * _r_cell[k] + _z_cell[k])];
            _vz[k] = crude.vz.data[int(crude.vz.stride * _r_cell[k] + _z_cell[k])];
            _vrz[k] = crude.vrz.data[int(crude.vrz.stride * _r_cell[k] + _z_cell[k])];

        }

        // Normalizing input rval and zval to the crude crude.
        float rnorm = (rval - crude.r.data[int(crude.r.stride * _r_cell[0] + _z_cell[0])]) / crude.dr;
        float znorm = (zval - crude.z.data[int(crude.z.stride * _r_cell[0] + _z_cell[0])]) / crude.dz;

        // Constructing our matrix consisting of function and partial derivative data
        _v.reshape(2, 2);
        _vr.reshape(2, 2);
        _vz.reshape(2, 2);
        _vrz.reshape(2, 2);

        _fval(0, 0) = _v(0, 0);
        _fval(0, 1) = _v(1, 0);
        _fval(1, 0) = _v(0, 1);
        _fval(1, 1) = _v(1, 1);

        _fval(2, 0) = _vr(0, 0);
        _fval(2, 1) = _vr(1, 0);
        _fval(3, 0) = _vr(0, 1);
        _fval(3, 1) = _vr(1, 1);

        _fval(0, 2) = _vz(0, 0);
        _fval(0, 3) = _vz(1, 0);
        _fval(1, 2) = _vz(0, 1);
        _fval(1, 3) = _vz(1, 1);

        _fval(2, 2) = _vrz(0, 0);
        _fval(2, 3) = _vrz(1, 0);
        _fval(3, 2) = _vrz(0, 1);
        _fval(3, 3) = _vrz(1, 1);

        // // compute f(r, z)
        if (data_tag == 0) {
            _r[0] = 1;
            _r[1] = rnorm;
            _r[2] = rnorm * rnorm;
            _r[3] = rnorm * rnorm * rnorm;

            _z[0] = 1;
            _z[1] = znorm;
            _z[2] = znorm * znorm;
            _z[3] = znorm * znorm * znorm;

        }

        // Data tag can be used to interpolate partial derivative data (to do).

        // // // // compute f_{r}(r, z)
        // // // else if (data_tag == 1) {
        // // //     temp_r_vector[0] = 0;
        // // //     temp_r_vector[1] = 1;
        // // //     temp_r_vector[2] = 2 * rval;
        // // //     temp_r_vector[3] = 3 * rval * rval;

        // // //     temp_z_vector[0] = 1;
        // // //     temp_z_vector[1] = zval;
        // // //     temp_z_vector[2] = zval * zval;
        // // //     temp_z_vector[3] = zval * zval * zval;
        // // // }

        // // // // compute f_{z}(r, z)
        // // // else if (data_tag == 2) {
        // // //     temp_r_vector[0] = 1;
        // // //     temp_r_vector[1] = rval;
        // // //     temp_r_vector[2] = rval * rval;
        // // //     temp_r_vector[3] = rval * rval * rval;

        // // //     temp_z_vector[0] = 0;
        // // //     temp_z_vector[1] = 1;
        // // //     temp_z_vector[2] = 2 * zval;
        // // //     temp_z_vector[3] = 3 * zval * zval;
        // // // }

        // Performing our series of matrix multiplications to obtain our interpolated valu
        psuedo_ND temp_m = pnd_mat_mul(_fval, crude.rhs_nd);
        temp_m = pnd_mat_mul(crude.lhs_nd, temp_m);
        temp_m = pnd_mat_mul(temp_m, _z);
        temp_m = pnd_mat_mul(_r, temp_m);

        // Setting interpolated value in fine grid
        fine.v(i, j) = temp_m[0];

        _r.release(); _z.release(); _fval.release(); _r_cell.release(); _z_cell.release();
        _v.release(); _vr.release(); _vz.release(); _vrz.release(); temp_m.release();

    }
}
