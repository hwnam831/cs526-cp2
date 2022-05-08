#include "vv.h"

#define WIDTH_C 262144
#define bidx (get_group_id(0))
#define bidy (get_group_id(1))
#define tidx (get_local_id(0))
#define tidy (get_local_id(1))
#define blockDimX 32
#define blockDimY 1
#define idx (bidx*blockDimX+tidx)
#define idy (bidy*blockDimY+tidy)
#define C(y,x) C[(x)]

__kernel void vv(__global float *A, __global float *B, __global float *C, int width) {

	float a;
	float b;

	for (int i=0; i<TILE; i++){
		a = A[idx*TILE+i];
		b = B[idx*TILE+i];
		C(0, idx*TILE+i) = a*b;
	}
	
}


