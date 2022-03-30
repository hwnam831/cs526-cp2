#ifndef _IMREGIONMAX_KERNEL_H_
#define _IMREGIONMAX_KERNEL_H_

#include <stdio.h>
#include "imregionmax.h"


#define MaxF(x,y) (x>y?x:y)

__device__ float cal(float* temp) {
	float result = 1;
	float max;
	max = MaxF(temp[0], temp[1]);
	max = MaxF(max, temp[2]);
	max = MaxF(max, temp[3]);
	max = MaxF(max, temp[5]);
	max = MaxF(max, temp[6]);
	max = MaxF(max, temp[7]);
	max = MaxF(max, temp[8]);

	result = max>temp[4]?0:1;
	return result;
}


#define COALESCED_NUM  16
#define A(y,x) A[(y)*WIDTH_A+(x)]
#define C(y,x) C[(y)*WIDTH_C+(x)]
#define blockDimX 16
#define blockDimY 16
#define idx (blockIdx.x*blockDimX+threadIdx.x)
#define idy (blockIdx.y*blockDimY+threadIdx.y)
__global__ void imregionmax_naive(float* A, float* C, int width)
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




#endif // #ifndef _IMREGIONMAX_KERNEL_H_
