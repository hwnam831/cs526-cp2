

#include "conv.h"

#define bidx (get_group_id(0))
#define bidy (get_group_id(1))
#define tidx (get_local_id(0))
#define tidy (get_local_id(1))
#define blockDimX 32
#define blockDimY 1
#define idx (bidx*blockDimX+tidx)
#define idy (bidy*blockDimY+tidy)
#define A(y,x) A[(y)*WIDTH_A+(x)]
#define B(y,x) B[(y)*WIDTH_B+(x)]
#define C(y,x) C[(y)*WIDTH_C+(x)]
__kernel void conv(__global float *A, __global float *B, __global float *C, int width, int height, int w, int h) {
	int i;
	int j;
	float sum = 0;
	for (j=0; j<h; j=j+1) {
		for (i=0; i<w; i=i+1) {
			float a;
			float b;
			a = A(idy+j, idx+i);
			b = B(j, i);
			sum += a*b;
		}
	}
	C(idy, idx) = sum;
}

// __kernel void conv(__global float * A, __global float * B, __global float * C, int width, int height, int w, int h)
// {
// 	__local float shared_B[32];
// 	__local float shared_A[64];
// 	int i;
// 	int j;
// 	float sum = 0;
// 	for (j=0; j<h; j=(j+1))
// 	{
// 		for (i=0; i<w; i=(i+32))
// 		{
// 			shared_A[(tidx+0)]=A(idy+j, idx+i);
// 			shared_A[(tidx+32)]=A(idy+j, idx+i+32);			

// 			int it_2;
// 			shared_B[(tidx+0)]=B(j, ((i+0)+tidx));
// 			barrier(CLK_LOCAL_MEM_FENCE);
			
// 			for (it_2=0; it_2<32; it_2=(it_2+1))
// 			{
// 				float a;
// 				float b;
// 				a=shared_A[tidx+it_2];
// 				b=shared_B[it_2];
// 				sum+=(a*b);
// 			}
// 			barrier(CLK_LOCAL_MEM_FENCE);
// 		}
// 	}
// 	{
// 		C(idy, idx)=sum;
// 	}
// }

// __kernel void conv(__global float * A, __global float * B, __global float * C, int width, int height, int w, int h)
// {
// 	__local float shared_B[32];
// 	__local float shared_A[64];
// 	int i;
// 	int j;
// 	float sum = 0;
// 	for (j=0; j<h; j=(j+1))
// 	{
// 		float temp_B = B(j, ((0+0)+tidx));
// 		float temp_A1 = A(idy+j, idx+0);
// 		float temp_A2 = A(idy+j, idx+0+32);
// 		for (i=0; i<w; i=(i+32))
// 		{
// 			shared_A[(tidx+0)]=temp_A1;
// 			shared_A[(tidx+32)]=temp_A2;			

// 			int it_2;
// 			shared_B[(tidx+0)]=temp_B;
// 			barrier(CLK_LOCAL_MEM_FENCE);
// 			if(i + 32 < w){
// 				temp_B = B(j, ((i+0)+tidx + 32));
// 				temp_A1 = A(idy+j, idx+i + 32);
// 				temp_A2 = A(idy+j, idx+i+32 + 32);
// 			}
			
// 			for (it_2=0; it_2<32; it_2=(it_2+1))
// 			{
// 				float a;
// 				float b;
// 				a=shared_A[tidx+it_2];
// 				b=shared_B[it_2];
// 				sum+=(a*b);
// 			}
// 			barrier(CLK_LOCAL_MEM_FENCE);
// 		}
// 	}
// 	{
// 		C(idy, idx)=sum;
// 	}
// }


