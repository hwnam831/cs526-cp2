
#ifndef _TR_KERNEL_H_
#define _TR_KERNEL_H_

#include <stdio.h>
#include "tr.h"



#define WIDTH_A WA
#define WIDTH_C WC


#define COALESCED_NUM  32
#define A(y,x) A[(y)*WIDTH_A+(x)]
#define C(y,x) C[(y)*WIDTH_C+(x)]
#define blockDimX 16
#define blockDimY 16
#define idx (blockIdx.x*blockDimX+threadIdx.x)
#define idy (blockIdx.y*blockDimY+threadIdx.y)
__global__ void transpose_naive(float *A, float *C, int width) {
	int i = 0;
	float sum = 0;

	sum = A(idx, idy);
	C(idy, idx) = sum;
}



#endif // #ifndef _TR_KERNEL_H_
