

#ifndef _MV_KERNEL_H_
#define _MV_KERNEL_H_

#include <stdio.h>
#include "mv.h"

#define WIDTH_A WA
#define WIDTH_B WB
#define WIDTH_C WC

#define COALESCED_NUM  16
#define A(y,x) A[(y)*WIDTH_A+(x)]
#define globalDimY 1
#define blockDimX 256
#define blockDimY 1
#define idx (blockIdx.x*blockDimX+threadIdx.x)
#define idy (blockIdx.y*blockDimY+threadIdx.y)
__global__ void mv_naive(float *A, float *B, float *C, int width) {
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


#define COALESCED_NUM 32
#define blockDimX 32
#define blockDimY 1
#define gridDimX (gridDim.x)
#define gridDimY (gridDim.y)
#define idx (blockIdx.x*blockDimX+threadIdx.x)
#define idy (blockIdx.y*blockDimY+threadIdx.y)
#define bidy (blockIdx.y)
#define bidx (blockIdx.x)
#define tidx (threadIdx.x)
#define tidy (threadIdx.y)
#define merger_y 1
#define coalesced_idy (bidy/(COALESCED_NUM/(merger_y*blockDimY))*COALESCED_NUM)
#define globalDimY 1
#define A(y,x) A[(y)*WIDTH_A+(x)]
__global__ void mv_coalesced(float * A, float * B, float * C, int width)
{
	__shared__ float shared_0[32][33];
	int i;
	float sum;
	sum=0;
	for (i=0; i<WIDTH_A; i=(i+32))
	{
		int it_1;
		int it_2;
		#pragma unroll
		for (it_2=0; it_2<32; it_2=(it_2+1))
		{
			shared_0[it_2][tidx]=A(((idx+(-1*tidx))+it_2), (i+tidx));
		}
		__syncthreads();
		#pragma unroll
		for (it_1=0; it_1<32; it_1=(it_1+1))
		{
			float a;
			float b;
			a=shared_0[tidx][it_1];
			b=B[(it_1+i)];
			sum+=(a*b);
		}
		__syncthreads();
	}
	{
		C[idx]=sum;
	}
}


#define COALESCED_NUM 16
#define blockDimX 32
#define blockDimY 1
#define gridDimX (gridDim.x)
#define gridDimY (gridDim.y)
#define idx (blockIdx.x*blockDimX+threadIdx.x)
#define idy (blockIdx.y*blockDimY+threadIdx.y)
#define bidy (blockIdx.y)
#define bidx (blockIdx.x)
#define tidx (threadIdx.x)
#define tidy (threadIdx.y)
#define merger_y 1
#define coalesced_idy (bidy/(COALESCED_NUM/(merger_y*blockDimY))*COALESCED_NUM)
#define globalDimY 1
#define A(y,x) A[(y)*WIDTH_A+(x)]
__global__ void mv_opt(float * A, float * B, float * C, int width)
{
	int ibidx;
	int ntidx;
	__shared__ float shared_1[16];
	__shared__ float shared_0[2][16][17];
	int i;
	float sum;
	ibidx=(tidx/16);
	ntidx=(tidx%16);
	sum=0;
	int it_4;
	int tmp_0;
	tmp_0=(bidx*16);
	for (i=0; i<WIDTH_A; i=(i+16))
	{
		it_4=((i+tmp_0)%WIDTH_A);
		int it_2;
		int it_3;
		#pragma unroll
		for (it_2=0; it_2<16; it_2=(it_2+1))
		{
			shared_0[ibidx][it_2][ntidx]=A(((idx+(( - 1)*ntidx))+it_2), (it_4+ntidx));
		}
		__syncthreads();
		if ((tidx<16))
		{
			shared_1[(tidx+0)]=B[((it_4+0)+tidx)];
		}
		__syncthreads();
		#pragma unroll
		for (it_3=0; it_3<16; it_3=(it_3+1))
		{
			float a;
			float b;
			a=shared_0[ibidx][ntidx][(it_3+0)];
			b=shared_1[it_3];
			sum+=(a*b);
		}
		__syncthreads();
		__syncthreads();
	}
	C[idx]=sum;
}



#endif // #ifndef _MV_KERNEL_H_
