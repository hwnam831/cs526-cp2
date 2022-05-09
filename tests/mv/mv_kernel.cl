#include "mv.h"

#define bidx (get_group_id(0))
#define bidy (get_group_id(1))
#define tidx (get_local_id(0))
#define tidy (get_local_id(1))
#define WIDTH_A WA
#define WIDTH_B WB
#define WIDTH_C WC

#define COALESCED_NUM  16
#define A(y,x) A[(y)*WIDTH_A+(x)]
#define globalDimY 1
#define blockDimX 32
#define blockDimY 1
#define idx (bidx*blockDimX+tidx)
#define idy (bidy*blockDimY+tidy)
__kernel void mv(__global float *A, __global float *B, __global float *C, int width) {
	int i;
	float sum;
	sum = 0;

	for (i=0; i<WIDTH_A; i=i+1) {
		float a;
		float b;
		a = A(idx, i);
		b = B[i];
		sum += a*b;
	}
	C[idx] = sum;
}

// __kernel void mv(__global float *A, __global float *B, __global float *C, int width) {
// 	int i, l, it_1;
// 	float sum;
// 	sum = 0;
// 	__local float shared_A[1024];
// 	__local float shared_B[32];

// 	for (i=0; i<WIDTH_A; i=i+32) {
// 		shared_B[tidx]=B[i+tidx];
// 		for (l=0; l<32; l=(l+1))
//     		shared_A[l*32+tidx]=A(((idx-tidx)+l), (i+tidx));
// 		barrier(CLK_LOCAL_MEM_FENCE);

// 		for (it_1=0; it_1<32; it_1=(it_1+1)){
// 			float a;
// 			float b;
// 			a = shared_A[tidx*32+it_1];
// 			b = shared_B[it_1];
// 			sum += a*b;
// 		}
// 		barrier(CLK_LOCAL_MEM_FENCE);
// 	}
// 	C[idx] = sum;
// }

// __kernel void mv(__global float *A, __global float *B, __global float *C, int width) {
// 	int i, l, it_1, it_2, it_3;
// 	float sum;
// 	sum = 0;
// 	__local float shared_A[1024];
// 	__local float shared_B[32];
// 	float temp_A[32];
// 	float temp_B = B[0+tidx];
// 	for (it_2=0; it_2<32; it_2=(it_2+1))
// 		temp_A[it_2] = A(((idx-tidx)+it_2), (0+tidx));
// 	it_3 = 0;
// 	for (i=0; i<WIDTH_A; i=i+32) {
		
// 		shared_B[tidx]=temp_B;
// 		for (l=0; l<32; l=(l+1))
//     		shared_A[l*32+tidx]=temp_A[l];
// 		barrier(CLK_LOCAL_MEM_FENCE);

// 		if(i+32 < WIDTH_A) {
// 			temp_B = B[i+tidx+32];
// 			for (it_2=0; it_2<32; it_2=(it_2+1))
// 				temp_A[it_2] = A(((idx-tidx)+it_2), (it_3*32+32+tidx)); // 32 is the step of outer loop for accessing A
// 		}
// 		for (it_1=0; it_1<32; it_1=(it_1+1)){
// 			float a;
// 			float b;
// 			a = shared_A[tidx*32+it_1];
// 			b = shared_B[it_1];
// 			sum += a*b;
// 		}
// 		barrier(CLK_LOCAL_MEM_FENCE);
// 		it_3 += 1;
// 	}
// 	C[idx] = sum;
// }

// __kernel void mv(__global float *A, __global float *B, __global float *C, int width) {
// 	int i, l, it_1, it_2, it_3;
// 	float sum;
// 	sum = 0;
// 	__local float shared_A[1024];
// 	__local float shared_B[32];
// 	float temp_A[32];
// 	float temp_B = B[0+tidx];
// 	for (it_2=0; it_2<32; it_2=(it_2+1))
// 		temp_A[it_2] = A(((idx-tidx)+it_2), (0+tidx));
// 	it_3 = 0;
// 	for (i=0; i<WIDTH_A; i=i+64) { // unroll by 2
		
// 		shared_B[tidx]=temp_B;
// 		for (l=0; l<32; l=(l+1))
//     		shared_A[l*32+tidx]=temp_A[l];
// 		barrier(CLK_LOCAL_MEM_FENCE);

// 		if(i+32 < WIDTH_A) {
// 			temp_B = B[i+tidx+32];
// 			for (it_2=0; it_2<32; it_2=(it_2+1))
// 				temp_A[it_2] = A(((idx-tidx)+it_2), (it_3*32+32+tidx)); // 32 is the step of outer loop for accessing A
// 		}
// 		for (it_1=0; it_1<32; it_1=(it_1+1)){
// 			float a;
// 			float b;
// 			a = shared_A[tidx*32+it_1];
// 			b = shared_B[it_1];
// 			sum += a*b;
// 		}
// 		barrier(CLK_LOCAL_MEM_FENCE);
// 		it_3 += 1;

// 		i = i + 32;
// 		shared_B[tidx]=temp_B;
// 		for (l=0; l<32; l=(l+1))
//     		shared_A[l*32+tidx]=temp_A[l];
// 		barrier(CLK_LOCAL_MEM_FENCE);

// 		if(i+32 < WIDTH_A) {
// 			temp_B = B[i+tidx+32];
// 			for (it_2=0; it_2<32; it_2=(it_2+1))
// 				temp_A[it_2] = A(((idx-tidx)+it_2), (it_3*32+32+tidx)); // 32 is the step of outer loop for accessing A
// 		}
// 		for (it_1=0; it_1<32; it_1=(it_1+1)){
// 			float a;
// 			float b;
// 			a = shared_A[tidx*32+it_1];
// 			b = shared_B[it_1];
// 			sum += a*b;
// 		}
// 		barrier(CLK_LOCAL_MEM_FENCE);
// 		it_3 += 1;
// 	}
// 	C[idx] = sum;
// }
// __kernel void matrixMul(__global float *A, __global float *B, __global float *C, int width, int height) {
// 	__local float shared_0[32];
// 	int i,j;
// 	float sum;
// 	sum = 0;
// 	for (i=0; i<width; i=(i+32)){
// 		int it_1;
// 		shared_0[(tidx+0)]=A(idy, (i+tidx));
// 		barrier(CLK_LOCAL_MEM_FENCE);

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

