
#ifndef _CONV_KERNEL_H_
#define _CONV_KERNEL_H_

#include <stdio.h>
#include "conv.h"

#define A(y,x) A[(y)*WIDTH_A+(x)]
#define B(y,x) B[(y)*WIDTH_B+(x)]
#define C(y,x) C[(y)*WIDTH_C+(x)]
#define blockDimX 16
#define blockDimY 16
#define idx (blockIdx.x*blockDimX+threadIdx.x)
#define idy (blockIdx.y*blockDimY+threadIdx.y)
__global__ void conv_naive(float *A, float *B, float *C, int width, int height, int w, int h) {
	int i;
	int j;
	float sum = 0;
	for (j=0; j<16; j=j+1) {
		for (i=0; i<16; i=i+1) {
			float a;
			float b;
			a = A(idy-j+h, idx-i+w);
			b = B(j, i);
			sum += a*b;
		}
	}
	C(idy, idx) = sum;
}




#endif // #ifndef _CONV_KERNEL_H_
