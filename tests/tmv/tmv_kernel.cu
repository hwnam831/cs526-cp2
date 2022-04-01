
#ifndef _TMV_KERNEL_H_
#define _TMV_KERNEL_H_

#include <stdio.h>
#include "tmv.h"

#define WIDTH_A WA

#define COALESCED_NUM  32
#define globalDimY 1
#define blockDimX 256
#define blockDimY 1
#define idx (blockIdx.x*blockDimX+threadIdx.x)
#define idy (blockIdx.y*blockDimY+threadIdx.y)
#define A(y,x) A[(y)*WIDTH_A+(x)]
__global__ void tmv_naive(float *A, float *B, float *C, int width) {
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


#endif // #ifndef _TMV_KERNEL_H_
