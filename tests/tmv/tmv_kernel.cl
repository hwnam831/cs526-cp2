#include "tmv.h"

#define WIDTH_A WA
#define bidx (get_group_id(0))
#define bidy (get_group_id(1))
#define tidx (get_local_id(0))
#define tidy (get_local_id(1))
#define globalDimY 1
#define blockDimX 256
#define blockDimY 1
#define idx (bidx*blockDimX+tidx)
#define idy (bidy*blockDimY+tidy)
#define A(y,x) A[(y)*WIDTH_A+(x)]
__kernel void tmv(__global float *A, __global float *B, __global float *C, int width) {
	int i;
	i = 0;
	float sum;
	sum = 0;

	for (i=0; i<width; i=i+1) {
		float a;
		float b;
		a = A(i, idx);
		b = B[i];
		sum += a*b;
	}
	C[idx] = sum;
}


