
#ifndef _TMV_KERNEL_H_
#define _TMV_KERNEL_H_

#include <stdio.h>
#include "tmv.h"

#define WIDTH_A WA

#define COALESCED_NUM  32
#define globalDimY 1
#define blockDimX 256
#define blockDimY 1
#define idx (blockIdx.x*blockDimX+threadIdx.x)
#define idy (blockIdx.y*blockDimY+threadIdx.y)
#define A(y,x) A[(y)*WIDTH_A+(x)]
__global__ void tmv_naive(float *A, float *B, float *C, int width) {
	int i;
	i = 0;
	float sum;
	sum = 0;

	for (i=0; i<width; i=i+1) {
		float a;
		float b;
		a = A(i, idx);
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
__global__ void tmv_coalesced(float * A, float * B, float * C, int width)
{
	__shared__ float shared_0[32];
	int i;
	float sum;
	i=0;
	sum=0;
	for (i=0; i<width; i=(i+32))
	{
		int it_1;
		shared_0[(tidx+0)]=B[(i+tidx)];
		__syncthreads();
		#pragma unroll
		for (it_1=0; it_1<32; it_1=(it_1+1))
		{
			float a;
			float b;
			a=A((it_1+i), idx);
			b=shared_0[it_1];
			sum+=(a*b);
		}
		__syncthreads();
	}
	{
		C[idx]=sum;
	}
}

#define COALESCED_NUM 32
#define blockDimX 512
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
__global__ void tmv_opt(float * A, float * B, float * C, int width)
{
	__shared__ float shared_0[32];
	int i;
	float sum;
	i=0;
	sum=0;
	for (i=0; i<width; i=(i+32))
	{
		int it_1;
		if ((tidx<32))
		{
			shared_0[(tidx+0)]=B[(i+tidx)];
		}
		__syncthreads();
		#pragma unroll
		for (it_1=0; it_1<32; it_1=(it_1+1))
		{
			float a;
			float b;
			a=A((it_1+i), idx);
			b=shared_0[it_1];
			sum+=(a*b);
		}
		__syncthreads();
	}
	{
		C[idx]=sum;
	}
}

#define COALESCED_NUM 32
#define blockDimX 512
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
__global__ void tmv_pref(float * A, float * B, float * C, int width)
{
	__shared__ float shared_0[32];
	int i;
	float sum;
	i=0;
	sum=0;
	float tmp_0;
	if ((tidx<32))
	{
		tmp_0=B[(0+tidx)];
	}
	for (i=0; i<width; i=(i+32))
	{
		int it_1;
		if ((tidx<32))
		{
			shared_0[(tidx+0)]=tmp_0;
		}
		__syncthreads();
		#pragma unroll
		for (it_1=0; it_1<32; it_1=(it_1+1))
		{
			float a;
			float b;
			a=A((it_1+i), idx);
			b=shared_0[it_1];
			sum+=(a*b);
		}
		if ((tidx<32))
		{
			if ((i<(width-32)))
			{
				tmp_0=B[((i+32)+tidx)];
			}
		}
		__syncthreads();
	}
	{
		C[idx]=sum;
	}
}


#endif // #ifndef _TMV_KERNEL_H_
