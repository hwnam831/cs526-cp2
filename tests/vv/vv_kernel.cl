#include "vv.h"

#define WIDTH_C WC
#define bidx (get_group_id(0))
#define bidy (get_group_id(1))
#define tidx (get_local_id(0))
#define tidy (get_local_id(1))
#define blockDimX 16
#define blockDimY 16
#define idx (bidx*blockDimX+tidx)
#define idy (bidy*blockDimY+tidy)
#define C(y,x) C[(y)*WIDTH_C+(x)]
__kernel void vv(__global float *A, __global float *B, __global float *C, int width) {

	float sum;
	float a;
	float b;
	sum = 0;
	a = A[idy];
	b = B[idx];
	sum += a*b;
	C(idy, idx)+=sum;
}


