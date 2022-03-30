

#ifndef _STRM_KERNEL_H_
#define _STRM_KERNEL_H_

#include <stdio.h>

#define INPUT_WIDTH 8192

#define WIDTH_A INPUT_WIDTH
#define WIDTH_B INPUT_WIDTH
#define WIDTH_C INPUT_WIDTH

#define A(y,x) A[(y)*WIDTH_A+(x)]
#define B(y,x) B[(y)*WIDTH_B+(x)]
#define C(y,x) C[(y)*WIDTH_C+(x)]
__global__ void matrix_naive(float *A, float *B, float *C, int width, int input_width)
{
    int tidx = threadIdx.x;
	int tidy = threadIdx.y;
	int idx = blockIdx.x * blockDim.x+tidx;
	int idy = blockIdx.y * blockDim.y+tidy;
	float sum = 0;
	for (int i=0; i<width; i++) {
		sum+=A(idy, i)*B(i, idx);
	}
	C(idy, idx) -= sum;
}




#endif // #ifndef _STRM_KERNEL_H_
