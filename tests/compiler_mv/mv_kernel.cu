

#ifndef _MV_KERNEL_H_
#define _MV_KERNEL_H_

#include <stdio.h>
#include "mv.h"

#define WIDTH_A WA
#define WIDTH_B WB
#define WIDTH_C WC

#define COALESCED_NUM  16
#define A(y,x) A[(y)*WIDTH_A+(x)]
#define globalDimY 1
#define blockDimX 256
#define blockDimY 1
#define idx (blockIdx.x*blockDimX+threadIdx.x)
#define idy (blockIdx.y*blockDimY+threadIdx.y)
__global__ void mv_naive(float *A, float *B, float *C, int width) {
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


#endif // #ifndef _MV_KERNEL_H_
