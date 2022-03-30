#include "tr.h"

#define WIDTH_A WA
#define WIDTH_C WC

#define bidx (get_group_id(0))
#define bidy (get_group_id(1))
#define tidx (get_local_id(0))
#define tidy (get_local_id(1))
#define A(y,x) A[(y)*WIDTH_A+(x)]
#define C(y,x) C[(y)*WIDTH_C+(x)]
#define blockDimX 16
#define blockDimY 16
#define idx (bidx*blockDimX+tidx)
#define idy (bidy*blockDimY+tidy)
__kernel void tr(__global float *A, __global float *C, int width) {
	int i = 0;
	float sum = 0;

	sum = A(idx, idy);
	C(idy, idx) = sum;
}


