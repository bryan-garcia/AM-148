#ifndef _CudaGrid_DRIVER_
#define _CudaGrid_DRIVER_

#include "CudaGrid.h"

extern "C" {
	void bicubic_main(float rmin, float rmax, float zmin, float zmax, \
        int nr_crude, int nz_crude, int nr_fine, int nz_fine, \
        float* crude_data, float* fine_data, int option);
}
#endif
