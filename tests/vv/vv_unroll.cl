#include "vv.h"

#define WIDTH_C WC
#define bidx (get_group_id(0))
#define bidy (get_group_id(1))
#define tidx (get_local_id(0))
#define tidy (get_local_id(1))
#define blockDimX 32
#define blockDimY 1
#define idx (bidx*blockDimX+tidx)
#define idy (bidy*blockDimY+tidy)
#define C(y,x) C[(y)*WIDTH_C+(x)]
/*
__kernel void vv(__global float *A, __global float *B, __global float *C, int width) {

	float a;
	float b;
	#pragma unroll 2
	for (int i=0; i<TILE; i++){
		a = A[idx*TILE+i];
		b = B[idx*TILE+i];
		C(0, idx*TILE+i) = a*b;
	}
	
}


__kernel void vv(__global float *A, __global float *B, __global float *C, int width) {

	float a;
	float b;
	int it, it_2;
	__local float shared_A[1024];
	__local float shared_B[1024];
	for (int i=0; i<TILE; i+=32){
		for(it_2=0; it_2<32; it_2++){
			shared_A[it_2*32+tidx] = A[(idx-tidx+it_2)*TILE + i+tidx];
			shared_B[it_2*32+tidx] = B[(idx-tidx+it_2)*TILE + i+tidx];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		#pragma unroll 2
		for(it=0; it<32; it++){
			a = shared_A[tidx*32+it];
			b = shared_B[tidx*32+it];
			C(0, idx*TILE+i+it) = a*b;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
}
*/

__kernel void vv(__global float *A, __global float *B, __global float *C, int width) {

	float a;
	float b;
	int it, it_2;
	__local float shared_A[1024];
	__local float shared_B[1024];
	float local_A[32];
	float local_B[32];
	for (it_2=0; it_2<32; it_2=(it_2+1)){
		local_A[it_2] = A[(idx-tidx+it_2)*TILE + tidx];
		local_B[it_2] = B[(idx-tidx+it_2)*TILE + tidx];
	}
	for (int i=0; i<TILE; i+=32){
		for(it_2=0; it_2<32; it_2++){
			shared_A[it_2*32+tidx] = local_A[it_2];
			shared_B[it_2*32+tidx] = local_B[it_2];
		}
		
		barrier(CLK_LOCAL_MEM_FENCE);
		if(i+32<TILE){
			for(it_2=0; it_2<32; it_2++){
				local_A[it_2] = A[(idx-tidx+it_2)*TILE + i+32+tidx];
				local_B[it_2] = B[(idx-tidx+it_2)*TILE + i+32+tidx];
			}
		}
		#pragma unroll 2
		for(it=0; it<32; it++){
			a = shared_A[tidx*32+it];
			b = shared_B[tidx*32+it];
			C(0, idx*TILE+i+it) = a*b;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
}

