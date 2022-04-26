#include "mv.h"

#define bidx (get_group_id(0))
#define bidy (get_group_id(1))
#define tidx (get_local_id(0))
#define tidy (get_local_id(1))
#define WIDTH_A WA
#define WIDTH_B WB
#define WIDTH_C WC

#define COALESCED_NUM  16
#define A(y,x) A[(y)*WIDTH_A+(x)]
#define globalDimY 1
#define blockDimX 32
#define blockDimY 1
#define idx (bidx*blockDimX+tidx)
#define idy (bidy*blockDimY+tidy)
__kernel void mv(__global float *A, __global float *B, __global float *C, int width) {
	int i;
	float sum;
	sum = 0;

	for (i=0; i<WIDTH_A; i=i+1) {
		float a;
		float b;
		a = A(idx, i);
		b = B[i];
		sum += a*b;
	}
	C[idx] = sum;
}