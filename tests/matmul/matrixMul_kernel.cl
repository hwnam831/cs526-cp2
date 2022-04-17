#include "matrixMul.h"

#define bidx (get_group_id(0))
#define bidy (get_group_id(1))
#define tidx (get_local_id(0))
#define tidy (get_local_id(1))
//#define blockDimX 16
#define blockDimX 32
//#define blockDimY 16
#define blockDimY 1
#define idx (bidx*blockDimX+tidx)
#define idy (bidy*blockDimY+tidy)
#define WIDTH_A WA
#define WIDTH_B WB
#define WIDTH_C WC
#define A(y,x) A[(y)*WIDTH_A+(x)]
#define B(y,x) B[(y)*WIDTH_B+(x)]
#define C(y,x) C[(y)*WIDTH_C+(x)]
// __kernel void matrixMul(__global float *A, __global float *B, __global float *C, int width, int height) {
// 	int i;
// 	float sum;
// 	sum = 0;
// 	for (i=0; i<width; i=i+1) {
// 		float a;
// 		float b;
// 		a = A(idy, i);
// 		b = B(i, idx);
// 		sum += a*b;
// 	}
// 	C(idy, idx) = sum;
// }

__kernel void matrixMul(__global float *A, __global float *B, __global float *C, int width, int height) {
	__local float shared_0[32];
	int i,j;
	float sum;
	sum = 0;
	for (i=0; i<width; i=(i+32)){
		int it_1;
		shared_0[(tidx+0)]=A(idy, (i+tidx));
		barrier(CLK_LOCAL_MEM_FENCE);

		for (it_1=0; it_1<32; it_1=(it_1+1)){
			float a;
			float b;
			a=shared_0[it_1];
			b=B((it_1+i), idx);
			sum += a*b;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	C(idy, idx) = sum;
}

// __kernel void matrixMul(__global float *A, __global float *B, __global float *C, int width, int height) {
// 	__local float shared_0[32];
// 	int i;
// 	float sum;
// 	sum = 0;
// 	float tmp = A(idy, ((0+tidx)+0));
// 	for (i=0; i<width; i=(i+32)){
// 		int it_1;
// 		shared_0[(tidx+0)]=tmp; 
// 		//shared_0[(tidx+0)]=A(idy, (i+tidx));
// 		barrier(CLK_LOCAL_MEM_FENCE);
// 		if (i+32<width) //bound check
// 			tmp = A(idy, (((i+32)+tidx)+0)); 
// 		for (it_1=0; it_1<32; it_1=(it_1+1)){
// 			float a;
// 			float b;
// 			a=shared_0[it_1];
// 			b=B((it_1+i), idx);
// 			sum += a*b;
// 		}
// 		barrier(CLK_LOCAL_MEM_FENCE);
// 	}
// 	C(idy, idx) = sum;
// }