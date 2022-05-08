#include "tr.h"

#define WIDTH_A WA
#define WIDTH_C WC

#define bidx (get_group_id(0))
#define bidy (get_group_id(1))
#define tidx (get_local_id(0))
#define tidy (get_local_id(1))
#define A(y,x) A[(y)*WIDTH_A+(x)]
#define C(y,x) C[(y)*WIDTH_C+(x)]
#define blockDimX 32
#define blockDimY 1
#define idx (bidx*blockDimX+tidx)
#define idy (bidy*blockDimY+tidy)
/*
__kernel void tr(__global float *A, __global float *C, int width) {
	int i = 0;
	float sum = 0;
	#pragma unroll 2
	for(i=0; i<TILE; i++){
		sum = A(idx, idy*TILE+i);
		C(idy*TILE+i, idx) = sum;
	}
}


__kernel void tr(__global float *A, __global float *C, int width) {
	int i, it, it_2;
	float sum = 0;
	__local float shared_A[1024];
	for(i=0; i<TILE; i+=32){
		for(it_2=0; it_2<32; it_2++){
			shared_A[it_2*32+tidx] = A(idx-tidx+it_2,idy*TILE+i+tidx);
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		#pragma unroll 2
		for(it=0; it<32; it++){
			sum = shared_A[tidx*32+it];
			C(idy*TILE+i+it, idx) = sum;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}
*/
__kernel void tr(__global float *A, __global float *C, int width) {
	int i, it, it_2;
	float sum = 0;
	__local float shared_A[1024];
	float local_A[32];
	for (it_2=0; it_2<32; it_2=(it_2+1))
		local_A[it_2] = A(((idx-tidx)+it_2), (idy*TILE+tidx));
	for(i=0; i<TILE; i+=32){

		for(it_2=0; it_2<32; it_2++)
			shared_A[it_2*32+tidx] = local_A[it_2];
		
		barrier(CLK_LOCAL_MEM_FENCE);
		if(i+32<TILE){
			for(it_2=0; it_2<32; it_2++){
				local_A[it_2] = A(idx-tidx+it_2,idy*TILE+i+tidx+32);
			}
		}
		
		#pragma unroll 2
		for(it=0; it<32; it++){
			sum = shared_A[tidx*32+it];
			C(idy*TILE+i+it, idx) = sum;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}