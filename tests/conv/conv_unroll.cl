

#include "conv.h"

#define bidx (get_group_id(0))
#define bidy (get_group_id(1))
#define tidx (get_local_id(0))
#define tidy (get_local_id(1))
#define blockDimX 32
#define blockDimY 1
#define idx (bidx*blockDimX+tidx)
#define idy (bidy*blockDimY+tidy)
#define A(y,x) A[(y)*WIDTH_A+(x)]
#define B(y,x) B[(y)*WIDTH_B+(x)]
#define C(y,x) C[(y)*WIDTH_C+(x)]
__kernel void conv(__global float *A, __global float *B, __global float *C, int width, int height, int w, int h) {
	int i;
	int j;
	float sum = 0;
	for (j=0; j<h; j=j+1) {
		#pragma unroll 2
		for (i=0; i<w; i=i+1) {
			float a;
			float b;
			a = A(idy+j, idx+i);
			b = B(j, i);
			sum += a*b;
		}
	}
	C(idy, idx) = sum;
}



