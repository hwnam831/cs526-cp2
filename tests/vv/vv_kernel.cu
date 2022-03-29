
#ifndef _VV_KERNEL_H_
#define _VV_KERNEL_H_

#include <stdio.h>

#include "vv.h"

#define WIDTH_C WC

#define COALESCED_NUM  32
#define blockDimX 16
#define blockDimY 16
#define idx (blockIdx.x*blockDimX+threadIdx.x)
#define idy (blockIdx.y*blockDimY+threadIdx.y)
#define C(y,x) C[(y)*WIDTH_C+(x)]
__global__ void vectormul_naive(float *A, float *B, float *C, int width) {

	float sum;
	float a;
	float b;
	sum = 0;
	a = A[idy];
	b = B[idx];
	sum += a*b;
	C(idy, idx)+=sum;
}


#endif // #ifndef _VV_KERNEL_H_
