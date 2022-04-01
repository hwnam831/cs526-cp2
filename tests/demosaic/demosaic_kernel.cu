#ifndef _DEMOSAIC_KERNEL_H_
#define _DEMOSAIC_KERNEL_H_

#include <stdio.h>
#include "demosaic.h"



__device__ float cal(float* temp) {
	float result;
	result = temp[4] + 0.25 * (temp[1] + temp[3] + temp[5]
			+ temp[7]);
	return result;
}

#define COALESCED_NUM  16
#define A(y,x) A[(y)*WIDTH_A+(x)]
#define C(y,x) C[(y)*WIDTH_C+(x)]
#define blockDimX 16
#define blockDimY 16
#define idx (blockIdx.x*blockDimX+threadIdx.x)
#define idy (blockIdx.y*blockDimY+threadIdx.y)
__global__ void demosaic_naive(float* A, float* C, int width)
{
	float temp[9];
    int t;
    int i;
    int j;
    t = 0;
    for(i=0; i<3; i=i+1) {
		for(j=0; j<3; j=j+1){
			float a;
			a = A((idy+16-i), (idx+16-j));
			temp[t] = a;
			t=t+1;
		}
    }

    C(idy, idx) = cal(temp);
}




#endif // #ifndef _DEMOSAIC_KERNEL_H_
