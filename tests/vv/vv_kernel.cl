#include "vv.h"

#define WIDTH_C WC
#define bidx (get_group_id(0))
#define bidy (get_group_id(1))
#define tidx (get_local_id(0))
#define tidy (get_local_id(1))
#define blockDimX 32
#define blockDimY 1
#define idx (bidx*blockDimX+tidx)
#define idy (bidy*blockDimY+tidy)
#define C(y,x) C[(y)*WIDTH_C+(x)]
#define TILE 32
__kernel void vv(__global float *A, __global float *B, __global float *C, int width) {

	float a;
	float b;
	float sum = 0;

	for (int i=0; i<TILE; i++){
		a = A[idx*TILE+i];
		b = B[idx*TILE+i];
		sum += a*b;
	}
	C(0, idx) += sum;
}


