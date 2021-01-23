#include "CudaGrid.h"

extern "C" {

	void bicubic_main(float rmin, float rmax, float zmin, float zmax, \
        int nr_crude, int nz_crude, int nr_fine, int nz_fine, \
        float* crude_data, float* fine_data, int option) {

                CudaGrid crude(rmin, rmax, zmin, zmax, nr_crude, nz_crude);
                CudaGrid fine(rmin, rmax, zmin, zmax, nz_fine, nz_fine);

                bicubic_test(crude, fine, option, true);

                memcpy(crude_data, crude.v.data, crude.v.buffer_size);
                memcpy(fine_data, fine.v.data, fine.v.buffer_size);

                crude.release();
                fine.release();
	}
		
}
