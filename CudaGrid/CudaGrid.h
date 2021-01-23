/*------------------------------------------------------------------
CudaGrid.h

    Description: Class for managing bicubic interpolation of a grid

------------------------------------------------------------------*/

#ifndef _CudaGrid_H_
#define _CudaGrid_H_

#define V 0
#define VR 1
#define VZ 2
#define VRZ 3

#define BLOCK_SIZE 32

#include <cmath>
#include "psuedo_ND.h"

class CudaGrid {

    public:

        // Operational status of data: HOST, DEVICE, FROM_HOST_TO_DEVICE.
        int mode;

        // Number of grid points in the R and Z directions.
        int nr;
        int nz;

        // Grid spacial dimensions.
        float rmin;
        float zmin;
        float rmax;
        float zmax;

        // Step size between grid points.
        float dr;
        float dz;
        
        // Meshgrid arrays for R and Z.
        psuedo_ND r;
        psuedo_ND z;

        // Future CudaGrid Psi values, partial derivative data, and cross
        // derivative data.
        psuedo_ND v;
        psuedo_ND vr;
        psuedo_ND vz;
        psuedo_ND vrz;

        // Contains coordinates of cell closest to (r, z) coordinate
        psuedo_ND r_nearest_cell;
        psuedo_ND z_nearest_cell;

        // Bicubic class for handling the bicubic interpolation should go here.
        // Had issues composing CudaGrid class with my Bicubic class, so I had to
        // just throw everything into CudaGrid... future work will figure out the
        // issue and thus provide better code modularity.
        // (I kept getting gpu segmentation faults)

        // NOTE: changing all data types from double to float fixed a TON of issues.
        // This discovery has come too late in the project timeline to recover my composition
        // of the bicubic class with my CudaGrid class.

        psuedo_ND lhs_nd;
        psuedo_ND rhs_nd;
        psuedo_ND center_nd;
        psuedo_ND r_vector;
        psuedo_ND z_vector;

        psuedo_ND v_cell;
        psuedo_ND vr_cell;
        psuedo_ND vz_cell;
        psuedo_ND vrz_cell;
        psuedo_ND fval_nd;


        __host__ __device__
        CudaGrid(float rmin, float rmax, float zmin, float zmax, int nr, int nz, int mode = HOST) 
            : r(nr, nz, mode), z(nr, nz, mode), v(nr, nz, mode), vr(nr, nz, mode),
              vz(nr, nz, mode), vrz(nr, nz, mode), 
              r_nearest_cell(4, 1, mode), z_nearest_cell(4, 1, mode),
              lhs_nd(4, 4, mode), rhs_nd(4, 4, mode), center_nd(4, 4, mode), 
              r_vector(4, 1, mode), z_vector(1, 4, mode), 
              v_cell(4, 1, mode), vr_cell(4, 1, mode), vz_cell(4, 1, mode),
              vrz_cell(4, 1, mode), fval_nd(4, 4, mode) {
            
            this->rmin = rmin;
            this->rmax = rmax;
            this->zmin = zmin;
            this->zmax = zmax;
            this->nr = nr;
            this->nz = nz;
            this->mode = mode;
            this->setup_grid_data();
            
        };

        // Compute partial derivatives of grid data.
        void compute_pderivs();

        // Find nearest cell associated with (r, z) coordinate.
        __host__ __device__
        void find_nearest_cell(float rval, float zval);

        __device__
        void find_nearest_cell_gpu(float rval, float zval, psuedo_ND& temp_r, psuedo_ND& temp_z);

        __host__
        float interp(float rval, float zval, psuedo_ND& to_fill, int data_tag);
        
        __host__ __device__
        void setup_grid_data();
        __host__ __device__
        void copy(const CudaGrid& efit_obj, int mode, bool verbose = VERBOSE_FLAG);
        __host__ __device__
        void load(const CudaGrid& efit_obj, int direction, bool verbose = VERBOSE_FLAG);
        __host__ __device__
        void release();


        // Bicubic stuff

        // Note: could not resolve segfault issues with these functions on the GPU.
        // This was before I changed all data types from double to float, however. 

        __host__ __device__
        void setup_ND_data();

        __host__ __device__
        void setup_coefficient_matrices();

        __host__ __device__
        void set_interp_data(psuedo_ND& fval_nd, psuedo_ND v, psuedo_ND vr, psuedo_ND vz, psuedo_ND vrz);

        __host__ __device__
        void setup_interp_data_gpu(psuedo_ND fval_nd, psuedo_ND v, psuedo_ND vr, psuedo_ND vz, psuedo_ND vrz);

        __host__ __device__
        float interp_point(float r, float z, psuedo_ND& fval_nd,\
            psuedo_ND v_data, psuedo_ND vr_data, psuedo_ND vz_data, psuedo_ND vrz_data, int mode = 0);

        __host__ __device__
        float compute_bicubic(psuedo_ND& fval_nd);

};

__device__
inline float test_f1(float r, float z) { return sin(M_PI * (r + 2 * z)) + cos(3 * M_PI * z); };

__device__
inline float test_f2(float r, float z) { return sin(M_PI * r) * cos(M_PI * z); };

__device__
inline float test_f3(float r, float z) { return exp(- (r * r) - (z * z)); };

__host__
void init_data(CudaGrid& host_grid, int option);

__global__
void init_data_kernel(CudaGrid in_grid, int option = 1);

__host__
void compute_pderivs(CudaGrid& host_grid, bool unit_spacing = true);

__global__
void kernel_compute_pderivs_all(CudaGrid in_grid, bool unit_spacing);

__host__
void bicubic_test(CudaGrid& crude_grid, CudaGrid& fine_grid, int option, bool unit_spacing);

__global__
void kernel_bicubic_init(int index_i, int index_j, CudaGrid crude, CudaGrid fine, int data_tag);


#endif