
#ifndef _REDUCTION_KERNEL_H_
#define _REDUCTION_KERNEL_H_

#include <stdio.h>

#include "reduction.h"


__global__ void reduction_naive(float* d_odata, float* d_idata, int num_elements)
{

	int idx = (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
	d_odata[idx] = d_idata[idx]+d_idata[idx+num_elements/2];

}

#define COALESCED_NUM 16
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
#define globalDimX 65536
#define globalDimY 1
__global__ void reduction_opt_0(float * A, int size, int segSize)
{
	#pragma	gCompiler	gValue	segSize	262144

	int k;
	float sum;
	int nidx;
	__shared__ float shared_0[512];
	nidx=((((tidx/16)*2048)+(idx&15))+((idx/512)*16));
	float tmp_2;
	float tmp_3;
	float tmp_0;
	float tmp_1;
	sum=0;
	for (k=0; k<size; k=(k+262144))
	{
		float r;
		r=A[(nidx+k)];
		sum+=r;
	}
	tmp_0=sum;
	__syncthreads();
	sum=0;
	for (k=0; k<size; k=(k+262144))
	{
		float r;
		r=A[((nidx+131072)+k)];
		sum+=r;
	}
	tmp_1=sum;
	__syncthreads();
	float a;
	float b;
	float c;
	a=tmp_0;
	b=tmp_1;
	c=(a+b);
	tmp_2=c;
	sum=0;
	for (k=0; k<size; k=(k+262144))
	{
		float r;
		r=A[((nidx+65536)+k)];
		sum+=r;
	}
	tmp_0=sum;
	__syncthreads();
	sum=0;
	for (k=0; k<size; k=(k+262144))
	{
		float r;
		r=A[(((nidx+65536)+131072)+k)];
		sum+=r;
	}
	tmp_1=sum;
	__syncthreads();
	a=tmp_0;
	b=tmp_1;
	c=(a+b);
	tmp_3=c;
	a=tmp_2;
	b=tmp_3;
	c=(a+b);
	shared_0[(tidx+0)]=c;
	__syncthreads();
	if ((nidx<32768))
	{
		float a;
		float b;
		float c;
		a=shared_0[(tidx+0)];
		b=shared_0[(tidx+256)];
		c=(a+b);
		shared_0[(tidx+0)]=c;
	}
	__syncthreads();
	if ((nidx<16384))
	{
		float a;
		float b;
		float c;
		a=shared_0[(tidx+0)];
		b=shared_0[(tidx+128)];
		c=(a+b);
		shared_0[(tidx+0)]=c;
	}
	__syncthreads();
	if ((nidx<8192))
	{
		float a;
		float b;
		float c;
		a=shared_0[(tidx+0)];
		b=shared_0[(tidx+64)];
		c=(a+b);
		shared_0[(tidx+0)]=c;
	}
	__syncthreads();
	if ((nidx<4096))
	{
		float a;
		float b;
		float c;
		a=shared_0[(tidx+0)];
		b=shared_0[(tidx+32)];
		c=(a+b);
		shared_0[(tidx+0)]=c;
	}
	__syncthreads();
	if ((nidx<2048))
	{
		float a;
		float b;
		float c;
		a=shared_0[(tidx+0)];
		b=shared_0[(tidx+16)];
		c=(a+b);
		{
			A[nidx]=c;
		}
	}
}


#define COALESCED_NUM 16
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
#define globalDimX 512
#define globalDimY 1
__global__ void reduction_opt_1(float * A, int size, int segSize)
{
	#pragma	gCompiler	gValue	segSize	262144

	__shared__ float shared_1[512];
	float tmp_4;
	float tmp_5;
	float a;
	float b;
	float c;
	{
		a=A[idx];
	}
	{
		b=A[(idx+((262144/128)/2))];
	}
	c=(a+b);
	tmp_4=c;
	{
		a=A[(idx+512)];
	}
	{
		b=A[((idx+512)+((262144/128)/2))];
	}
	c=(a+b);
	tmp_5=c;
	a=tmp_4;
	b=tmp_5;
	c=(a+b);
	shared_1[(tidx+0)]=c;
	__syncthreads();
	if ((idx<256))
	{
		float a;
		float b;
		float c;
		a=shared_1[(tidx+0)];
		b=shared_1[(tidx+256)];
		c=(a+b);
		shared_1[(tidx+0)]=c;
	}
	__syncthreads();
	if ((idx<128))
	{
		float a;
		float b;
		float c;
		a=shared_1[(tidx+0)];
		b=shared_1[(tidx+128)];
		c=(a+b);
		shared_1[(tidx+0)]=c;
	}
	__syncthreads();
	if ((idx<64))
	{
		float a;
		float b;
		float c;
		a=shared_1[(tidx+0)];
		b=shared_1[(tidx+64)];
		c=(a+b);
		shared_1[(tidx+0)]=c;
	}
	__syncthreads();
	if ((idx<32))
	{
		float a;
		float b;
		float c;
		a=shared_1[(tidx+0)];
		b=shared_1[(tidx+32)];
		c=(a+b);
		shared_1[(tidx+0)]=c;
	}
	__syncthreads();
	if ((idx<16))
	{
		float a;
		float b;
		float c;
		a=shared_1[(tidx+0)];
		b=shared_1[(tidx+16)];
		c=(a+b);
		shared_1[(tidx+0)]=c;
	}
	__syncthreads();
	if ((idx<8))
	{
		float a;
		float b;
		float c;
		a=shared_1[(tidx+0)];
		b=shared_1[(tidx+8)];
		c=(a+b);
		shared_1[(tidx+0)]=c;
	}
	__syncthreads();
	if ((idx<4))
	{
		float a;
		float b;
		float c;
		a=shared_1[(tidx+0)];
		b=shared_1[(tidx+4)];
		c=(a+b);
		shared_1[(tidx+0)]=c;
	}
	__syncthreads();
	if ((idx<2))
	{
		float a;
		float b;
		float c;
		a=shared_1[(tidx+0)];
		b=shared_1[(tidx+2)];
		c=(a+b);
		shared_1[(tidx+0)]=c;
	}
	__syncthreads();
	if ((idx<1))
	{
		float a;
		float b;
		float c;
		a=shared_1[(tidx+0)];
		b=shared_1[(tidx+1)];
		c=(a+b);
		{
			A[idx]=c;
		}
	}
}



#define COALESCED_NUM 16
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
#define globalDimX 65536
#define globalDimY 1
__global__ void reduction_complex_opt_0(float * A, float * B, int size, int segSize)
{
	#pragma	gCompiler	gValue	segSize	262144

	int k;
	float sum;
	int nidx;
	__shared__ float shared_0[512];
	nidx=((((tidx/16)*2048)+(idx&15))+((idx/512)*16));
	float tmp_4;
	float tmp_5;
	float tmp_2;
	float tmp_3;
	sum=0;
	for (k=0; k<size; k=(k+262144))
	{
		float real;
		float img;
		struct float2 * tmp_0;
		struct float2 tmp_1;
		tmp_0=((struct float2 * )A);
		tmp_1=tmp_0[(nidx+k)];
		real=tmp_1.x;
		img=tmp_1.y;
		sum+=real;
		sum+=img;
	}
	tmp_2=sum;
	__syncthreads();
	sum=0;
	for (k=0; k<size; k=(k+262144))
	{
		float real;
		float img;
		struct float2 * tmp_0;
		struct float2 tmp_1;
		tmp_0=((struct float2 * )A);
		tmp_1=tmp_0[((nidx+131072)+k)];
		real=tmp_1.x;
		img=tmp_1.y;
		sum+=real;
		sum+=img;
	}
	tmp_3=sum;
	__syncthreads();
	float a;
	float b;
	float c;
	a=tmp_2;
	b=tmp_3;
	c=(a+b);
	tmp_4=c;
	sum=0;
	for (k=0; k<size; k=(k+262144))
	{
		float real;
		float img;
		struct float2 * tmp_0;
		struct float2 tmp_1;
		tmp_0=((struct float2 * )A);
		tmp_1=tmp_0[((nidx+65536)+k)];
		real=tmp_1.x;
		img=tmp_1.y;
		sum+=real;
		sum+=img;
	}
	tmp_2=sum;
	__syncthreads();
	sum=0;
	for (k=0; k<size; k=(k+262144))
	{
		float real;
		float img;
		struct float2 * tmp_0;
		struct float2 tmp_1;
		tmp_0=((struct float2 * )A);
		tmp_1=tmp_0[(((nidx+65536)+131072)+k)];
		real=tmp_1.x;
		img=tmp_1.y;
		sum+=real;
		sum+=img;
	}
	tmp_3=sum;
	__syncthreads();
	a=tmp_2;
	b=tmp_3;
	c=(a+b);
	tmp_5=c;
	a=tmp_4;
	b=tmp_5;
	c=(a+b);
	shared_0[(tidx+0)]=c;
	__syncthreads();
	if ((nidx<32768))
	{
		float a;
		float b;
		float c;
		a=shared_0[(tidx+0)];
		b=shared_0[(tidx+256)];
		c=(a+b);
		shared_0[(tidx+0)]=c;
	}
	__syncthreads();
	if ((nidx<16384))
	{
		float a;
		float b;
		float c;
		a=shared_0[(tidx+0)];
		b=shared_0[(tidx+128)];
		c=(a+b);
		shared_0[(tidx+0)]=c;
	}
	__syncthreads();
	if ((nidx<8192))
	{
		float a;
		float b;
		float c;
		a=shared_0[(tidx+0)];
		b=shared_0[(tidx+64)];
		c=(a+b);
		shared_0[(tidx+0)]=c;
	}
	__syncthreads();
	if ((nidx<4096))
	{
		float a;
		float b;
		float c;
		a=shared_0[(tidx+0)];
		b=shared_0[(tidx+32)];
		c=(a+b);
		shared_0[(tidx+0)]=c;
	}
	__syncthreads();
	if ((nidx<2048))
	{
		float a;
		float b;
		float c;
		a=shared_0[(tidx+0)];
		b=shared_0[(tidx+16)];
		c=(a+b);
		{
			B[nidx]=c;
		}
	}
}


#define COALESCED_NUM 16
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
#define globalDimX 512
#define globalDimY 1
__global__ void reduction_complex_opt_1(float * A, float * B, int size, int segSize)
{
	#pragma	gCompiler	gValue	segSize	262144

	__shared__ float shared_1[512];
	float tmp_6;
	float tmp_7;
	float a;
	float b;
	float c;
	{
		a=B[idx];
	}
	{
		b=B[(idx+((262144/128)/2))];
	}
	c=(a+b);
	tmp_6=c;
	{
		a=B[(idx+512)];
	}
	{
		b=B[((idx+512)+((262144/128)/2))];
	}
	c=(a+b);
	tmp_7=c;
	a=tmp_6;
	b=tmp_7;
	c=(a+b);
	shared_1[(tidx+0)]=c;
	__syncthreads();
	if ((idx<256))
	{
		float a;
		float b;
		float c;
		a=shared_1[(tidx+0)];
		b=shared_1[(tidx+256)];
		c=(a+b);
		shared_1[(tidx+0)]=c;
	}
	__syncthreads();
	if ((idx<128))
	{
		float a;
		float b;
		float c;
		a=shared_1[(tidx+0)];
		b=shared_1[(tidx+128)];
		c=(a+b);
		shared_1[(tidx+0)]=c;
	}
	__syncthreads();
	if ((idx<64))
	{
		float a;
		float b;
		float c;
		a=shared_1[(tidx+0)];
		b=shared_1[(tidx+64)];
		c=(a+b);
		shared_1[(tidx+0)]=c;
	}
	__syncthreads();
	if ((idx<32))
	{
		float a;
		float b;
		float c;
		a=shared_1[(tidx+0)];
		b=shared_1[(tidx+32)];
		c=(a+b);
		shared_1[(tidx+0)]=c;
	}
	__syncthreads();
	if ((idx<16))
	{
		float a;
		float b;
		float c;
		a=shared_1[(tidx+0)];
		b=shared_1[(tidx+16)];
		c=(a+b);
		shared_1[(tidx+0)]=c;
	}
	__syncthreads();
	if ((idx<8))
	{
		float a;
		float b;
		float c;
		a=shared_1[(tidx+0)];
		b=shared_1[(tidx+8)];
		c=(a+b);
		shared_1[(tidx+0)]=c;
	}
	__syncthreads();
	if ((idx<4))
	{
		float a;
		float b;
		float c;
		a=shared_1[(tidx+0)];
		b=shared_1[(tidx+4)];
		c=(a+b);
		shared_1[(tidx+0)]=c;
	}
	__syncthreads();
	if ((idx<2))
	{
		float a;
		float b;
		float c;
		a=shared_1[(tidx+0)];
		b=shared_1[(tidx+2)];
		c=(a+b);
		shared_1[(tidx+0)]=c;
	}
	__syncthreads();
	if ((idx<1))
	{
		float a;
		float b;
		float c;
		a=shared_1[(tidx+0)];
		b=shared_1[(tidx+1)];
		c=(a+b);
		{
			B[idx]=c;
		}
	}
}


#endif // #ifndef _REDUCTION_KERNEL_H_
