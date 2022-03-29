

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "matrixMul.h"

#define WIDTH_A WA
#define WIDTH_B WB
#define WIDTH_C WC


////////////////////////////////////////////////////////////////////////////////
//! Matrix multiplication on the device: C = A * B
//! wA is A's width and wB is B's width
////////////////////////////////////////////////////////////////////////////////
#define A(y,x) A[(y)*WIDTH_A+(x)]
#define B(y,x) B[(y)*WIDTH_B+(x)]
#define C(y,x) C[(y)*WIDTH_C+(x)]
#define blockDimX 16
#define blockDimY 16
#define idx (blockIdx.x*blockDimX+threadIdx.x)
#define idy (blockIdx.y*blockDimY+threadIdx.y)
__global__ void matmul_naive(float *A, float *B, float *C, int width, int height) {
	int i;
	float sum;
	sum = 0;
	for (i=0; i<width; i=i+1) {
		float a;
		float b;
		a = A(idy, i);
		b = B(i, idx);
		sum += a*b;
	}
	C(idy, idx) = sum;
}


#endif // #ifndef _MATRIXMUL_KERNEL_H_
