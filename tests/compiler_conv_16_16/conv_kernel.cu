
#ifndef _CONV_KERNEL_H_
#define _CONV_KERNEL_H_

#include <stdio.h>
#include "conv.h"

#define A(y,x) A[(y)*WIDTH_A+(x)]
#define B(y,x) B[(y)*WIDTH_B+(x)]
#define C(y,x) C[(y)*WIDTH_C+(x)]
#define blockDimX 16
#define blockDimY 16
#define idx (blockIdx.x*blockDimX+threadIdx.x)
#define idy (blockIdx.y*blockDimY+threadIdx.y)
__global__ void conv_naive(float *A, float *B, float *C, int width, int height, int w, int h) {
	int i;
	int j;
	float sum = 0;
	for (j=0; j<16; j=j+1) {
		for (i=0; i<16; i=i+1) {
			float a;
			float b;
			a = A(idy-j+h, idx-i+w);
			b = B(j, i);
			sum += a*b;
		}
	}
	C(idy, idx) = sum;
}




#define COALESCED_NUM 16
#define blockDimX 256
#define blockDimY 1
#define gridDimX (gridDim.x)
#define gridDimY (gridDim.y)
#define idx (blockIdx.x*blockDimX+threadIdx.x)
#define idy (blockIdx.y*blockDimY+threadIdx.y)
#define bidy (blockIdx.y)
#define bidx (blockIdx.x)
#define tidx (threadIdx.x)
#define tidy (threadIdx.y)
#define merger_y 8
#define coalesced_idy (bidy/(COALESCED_NUM/(merger_y*blockDimY))*COALESCED_NUM)
#define A(y,x) A[(y)*WIDTH_A+(x)]
#define B(y,x) B[(y)*WIDTH_B+(x)]
#define C(y,x) C[(y)*WIDTH_C+(x)]
__global__ void conv_opt(float * A, float * B, float * C, int width, int height, int w, int h)
{
	__shared__ float shared_1[16][9];
	__shared__ float shared_0[272];
	int j;
	float sum_0 = 0;
	float sum_1 = 0;
	float sum_2 = 0;
	float sum_3 = 0;
	float sum_4 = 0;
	float sum_5 = 0;
	float sum_6 = 0;
	float sum_7 = 0;
	int it_2;
	for (j=0; j<(h-7); j=(j+1))
	{
		int it_2;
		if ((tidx<16))
		{
			shared_0[(tidx+0)]=A((((idy*8)+(( - 1)*j))+h), (idx+(( - 1)*0)));
		}
		shared_0[(tidx+16)]=A((((idy*8)+(( - 1)*j))+h), ((idx+(( - 1)*0))+16));
		__syncthreads();
		if ((tidx<16))
		{
			shared_1[(tidx+0)][0]=B((j+0), (0+tidx));
			shared_1[(tidx+0)][1]=B((j+1), (0+tidx));
			shared_1[(tidx+0)][2]=B((j+2), (0+tidx));
			shared_1[(tidx+0)][3]=B((j+3), (0+tidx));
			shared_1[(tidx+0)][4]=B((j+4), (0+tidx));
			shared_1[(tidx+0)][5]=B((j+5), (0+tidx));
			shared_1[(tidx+0)][6]=B((j+6), (0+tidx));
			shared_1[(tidx+0)][7]=B((j+7), (0+tidx));
		}
		__syncthreads();
		#pragma unroll
		for (it_2=0; it_2<16; it_2=(it_2+1))
		{
			float a;
			float b_0;
			float b_1;
			float b_2;
			float b_3;
			float b_4;
			float b_5;
			float b_6;
			float b_7;
			a=shared_0[((tidx+(( - 1)*(it_2+0)))+16)];
			b_0=shared_1[it_2][0];
			b_1=shared_1[it_2][1];
			b_2=shared_1[it_2][2];
			b_3=shared_1[it_2][3];
			b_4=shared_1[it_2][4];
			b_5=shared_1[it_2][5];
			b_6=shared_1[it_2][6];
			b_7=shared_1[it_2][7];
			sum_0+=(a*b_0);
			sum_1+=(a*b_1);
			sum_2+=(a*b_2);
			sum_3+=(a*b_3);
			sum_4+=(a*b_4);
			sum_5+=(a*b_5);
			sum_6+=(a*b_6);
			sum_7+=(a*b_7);
		}
		__syncthreads();
		__syncthreads();
	}
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*8)+(( - 1)*(h-1)))+h), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*8)+(( - 1)*(h-1)))+h), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_1[(tidx+0)][0]=B((h-1), (0+tidx));
		}
	}
	__syncthreads();
	#pragma unroll
	for (it_2=0; it_2<16; it_2=(it_2+1))
	{
		float a;
		float b_0;
		a=shared_0[((tidx+(( - 1)*(it_2+0)))+16)];
		b_0=shared_1[it_2][0];
		sum_0+=(a*b_0);
	}
	__syncthreads();
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*8)+(( - 1)*(h-2)))+h), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*8)+(( - 1)*(h-2)))+h), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_1[(tidx+0)][0]=B((h-2), (0+tidx));
		}
		{
			shared_1[(tidx+0)][1]=B((h-1), (0+tidx));
		}
	}
	__syncthreads();
	#pragma unroll
	for (it_2=0; it_2<16; it_2=(it_2+1))
	{
		float a;
		float b_0;
		float b_1;
		a=shared_0[((tidx+(( - 1)*(it_2+0)))+16)];
		b_0=shared_1[it_2][0];
		b_1=shared_1[it_2][1];
		sum_0+=(a*b_0);
		sum_1+=(a*b_1);
	}
	__syncthreads();
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*8)+(( - 1)*(h-3)))+h), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*8)+(( - 1)*(h-3)))+h), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_1[(tidx+0)][0]=B((h-3), (0+tidx));
		}
		{
			shared_1[(tidx+0)][1]=B((h-2), (0+tidx));
		}
		{
			shared_1[(tidx+0)][2]=B((h-1), (0+tidx));
		}
	}
	__syncthreads();
	#pragma unroll
	for (it_2=0; it_2<16; it_2=(it_2+1))
	{
		float a;
		float b_0;
		float b_1;
		float b_2;
		a=shared_0[((tidx+(( - 1)*(it_2+0)))+16)];
		b_0=shared_1[it_2][0];
		b_1=shared_1[it_2][1];
		b_2=shared_1[it_2][2];
		sum_0+=(a*b_0);
		sum_1+=(a*b_1);
		sum_2+=(a*b_2);
	}
	__syncthreads();
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*8)+(( - 1)*(h-4)))+h), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*8)+(( - 1)*(h-4)))+h), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_1[(tidx+0)][0]=B((h-4), (0+tidx));
		}
		{
			shared_1[(tidx+0)][1]=B((h-3), (0+tidx));
		}
		{
			shared_1[(tidx+0)][2]=B((h-2), (0+tidx));
		}
		{
			shared_1[(tidx+0)][3]=B((h-1), (0+tidx));
		}
	}
	__syncthreads();
	#pragma unroll
	for (it_2=0; it_2<16; it_2=(it_2+1))
	{
		float a;
		float b_0;
		float b_1;
		float b_2;
		float b_3;
		a=shared_0[((tidx+(( - 1)*(it_2+0)))+16)];
		b_0=shared_1[it_2][0];
		b_1=shared_1[it_2][1];
		b_2=shared_1[it_2][2];
		b_3=shared_1[it_2][3];
		sum_0+=(a*b_0);
		sum_1+=(a*b_1);
		sum_2+=(a*b_2);
		sum_3+=(a*b_3);
	}
	__syncthreads();
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*8)+(( - 1)*(h-5)))+h), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*8)+(( - 1)*(h-5)))+h), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_1[(tidx+0)][0]=B((h-5), (0+tidx));
		}
		{
			shared_1[(tidx+0)][1]=B((h-4), (0+tidx));
		}
		{
			shared_1[(tidx+0)][2]=B((h-3), (0+tidx));
		}
		{
			shared_1[(tidx+0)][3]=B((h-2), (0+tidx));
		}
		{
			shared_1[(tidx+0)][4]=B((h-1), (0+tidx));
		}
	}
	__syncthreads();
	#pragma unroll
	for (it_2=0; it_2<16; it_2=(it_2+1))
	{
		float a;
		float b_0;
		float b_1;
		float b_2;
		float b_3;
		float b_4;
		a=shared_0[((tidx+(( - 1)*(it_2+0)))+16)];
		b_0=shared_1[it_2][0];
		b_1=shared_1[it_2][1];
		b_2=shared_1[it_2][2];
		b_3=shared_1[it_2][3];
		b_4=shared_1[it_2][4];
		sum_0+=(a*b_0);
		sum_1+=(a*b_1);
		sum_2+=(a*b_2);
		sum_3+=(a*b_3);
		sum_4+=(a*b_4);
	}
	__syncthreads();
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*8)+(( - 1)*(h-6)))+h), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*8)+(( - 1)*(h-6)))+h), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_1[(tidx+0)][0]=B((h-6), (0+tidx));
		}
		{
			shared_1[(tidx+0)][1]=B((h-5), (0+tidx));
		}
		{
			shared_1[(tidx+0)][2]=B((h-4), (0+tidx));
		}
		{
			shared_1[(tidx+0)][3]=B((h-3), (0+tidx));
		}
		{
			shared_1[(tidx+0)][4]=B((h-2), (0+tidx));
		}
		{
			shared_1[(tidx+0)][5]=B((h-1), (0+tidx));
		}
	}
	__syncthreads();
	#pragma unroll
	for (it_2=0; it_2<16; it_2=(it_2+1))
	{
		float a;
		float b_0;
		float b_1;
		float b_2;
		float b_3;
		float b_4;
		float b_5;
		a=shared_0[((tidx+(( - 1)*(it_2+0)))+16)];
		b_0=shared_1[it_2][0];
		b_1=shared_1[it_2][1];
		b_2=shared_1[it_2][2];
		b_3=shared_1[it_2][3];
		b_4=shared_1[it_2][4];
		b_5=shared_1[it_2][5];
		sum_0+=(a*b_0);
		sum_1+=(a*b_1);
		sum_2+=(a*b_2);
		sum_3+=(a*b_3);
		sum_4+=(a*b_4);
		sum_5+=(a*b_5);
	}
	__syncthreads();
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*8)+(( - 1)*(h-7)))+h), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*8)+(( - 1)*(h-7)))+h), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_1[(tidx+0)][0]=B((h-7), (0+tidx));
		}
		{
			shared_1[(tidx+0)][1]=B((h-6), (0+tidx));
		}
		{
			shared_1[(tidx+0)][2]=B((h-5), (0+tidx));
		}
		{
			shared_1[(tidx+0)][3]=B((h-4), (0+tidx));
		}
		{
			shared_1[(tidx+0)][4]=B((h-3), (0+tidx));
		}
		{
			shared_1[(tidx+0)][5]=B((h-2), (0+tidx));
		}
		{
			shared_1[(tidx+0)][6]=B((h-1), (0+tidx));
		}
	}
	__syncthreads();
	#pragma unroll
	for (it_2=0; it_2<16; it_2=(it_2+1))
	{
		float a;
		float b_0;
		float b_1;
		float b_2;
		float b_3;
		float b_4;
		float b_5;
		float b_6;
		a=shared_0[((tidx+(( - 1)*(it_2+0)))+16)];
		b_0=shared_1[it_2][0];
		b_1=shared_1[it_2][1];
		b_2=shared_1[it_2][2];
		b_3=shared_1[it_2][3];
		b_4=shared_1[it_2][4];
		b_5=shared_1[it_2][5];
		b_6=shared_1[it_2][6];
		sum_0+=(a*b_0);
		sum_1+=(a*b_1);
		sum_2+=(a*b_2);
		sum_3+=(a*b_3);
		sum_4+=(a*b_4);
		sum_5+=(a*b_5);
		sum_6+=(a*b_6);
	}
	{
		C(((idy*8)+0), idx)=sum_0;
	}
	__syncthreads();
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*8)+(( - 1)*(0-1)))+h), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*8)+(( - 1)*(0-1)))+h), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_1[(tidx+0)][1]=B(0, (0+tidx));
		}
		{
			shared_1[(tidx+0)][2]=B(1, (0+tidx));
		}
		{
			shared_1[(tidx+0)][3]=B(2, (0+tidx));
		}
		{
			shared_1[(tidx+0)][4]=B(3, (0+tidx));
		}
		{
			shared_1[(tidx+0)][5]=B(4, (0+tidx));
		}
		{
			shared_1[(tidx+0)][6]=B(5, (0+tidx));
		}
		{
			shared_1[(tidx+0)][7]=B(6, (0+tidx));
		}
	}
	__syncthreads();
	#pragma unroll
	for (it_2=0; it_2<16; it_2=(it_2+1))
	{
		float a;
		float b_1;
		float b_2;
		float b_3;
		float b_4;
		float b_5;
		float b_6;
		float b_7;
		a=shared_0[((tidx+(( - 1)*(it_2+0)))+16)];
		b_1=shared_1[it_2][1];
		b_2=shared_1[it_2][2];
		b_3=shared_1[it_2][3];
		b_4=shared_1[it_2][4];
		b_5=shared_1[it_2][5];
		b_6=shared_1[it_2][6];
		b_7=shared_1[it_2][7];
		sum_1+=(a*b_1);
		sum_2+=(a*b_2);
		sum_3+=(a*b_3);
		sum_4+=(a*b_4);
		sum_5+=(a*b_5);
		sum_6+=(a*b_6);
		sum_7+=(a*b_7);
	}
	{
		C(((idy*8)+1), idx)=sum_1;
	}
	__syncthreads();
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*8)+(( - 1)*(0-2)))+h), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*8)+(( - 1)*(0-2)))+h), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_1[(tidx+0)][2]=B(0, (0+tidx));
		}
		{
			shared_1[(tidx+0)][3]=B(1, (0+tidx));
		}
		{
			shared_1[(tidx+0)][4]=B(2, (0+tidx));
		}
		{
			shared_1[(tidx+0)][5]=B(3, (0+tidx));
		}
		{
			shared_1[(tidx+0)][6]=B(4, (0+tidx));
		}
		{
			shared_1[(tidx+0)][7]=B(5, (0+tidx));
		}
	}
	__syncthreads();
	#pragma unroll
	for (it_2=0; it_2<16; it_2=(it_2+1))
	{
		float a;
		float b_2;
		float b_3;
		float b_4;
		float b_5;
		float b_6;
		float b_7;
		a=shared_0[((tidx+(( - 1)*(it_2+0)))+16)];
		b_2=shared_1[it_2][2];
		b_3=shared_1[it_2][3];
		b_4=shared_1[it_2][4];
		b_5=shared_1[it_2][5];
		b_6=shared_1[it_2][6];
		b_7=shared_1[it_2][7];
		sum_2+=(a*b_2);
		sum_3+=(a*b_3);
		sum_4+=(a*b_4);
		sum_5+=(a*b_5);
		sum_6+=(a*b_6);
		sum_7+=(a*b_7);
	}
	{
		C(((idy*8)+2), idx)=sum_2;
	}
	__syncthreads();
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*8)+(( - 1)*(0-3)))+h), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*8)+(( - 1)*(0-3)))+h), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_1[(tidx+0)][3]=B(0, (0+tidx));
		}
		{
			shared_1[(tidx+0)][4]=B(1, (0+tidx));
		}
		{
			shared_1[(tidx+0)][5]=B(2, (0+tidx));
		}
		{
			shared_1[(tidx+0)][6]=B(3, (0+tidx));
		}
		{
			shared_1[(tidx+0)][7]=B(4, (0+tidx));
		}
	}
	__syncthreads();
	#pragma unroll
	for (it_2=0; it_2<16; it_2=(it_2+1))
	{
		float a;
		float b_3;
		float b_4;
		float b_5;
		float b_6;
		float b_7;
		a=shared_0[((tidx+(( - 1)*(it_2+0)))+16)];
		b_3=shared_1[it_2][3];
		b_4=shared_1[it_2][4];
		b_5=shared_1[it_2][5];
		b_6=shared_1[it_2][6];
		b_7=shared_1[it_2][7];
		sum_3+=(a*b_3);
		sum_4+=(a*b_4);
		sum_5+=(a*b_5);
		sum_6+=(a*b_6);
		sum_7+=(a*b_7);
	}
	{
		C(((idy*8)+3), idx)=sum_3;
	}
	__syncthreads();
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*8)+(( - 1)*(0-4)))+h), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*8)+(( - 1)*(0-4)))+h), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_1[(tidx+0)][4]=B(0, (0+tidx));
		}
		{
			shared_1[(tidx+0)][5]=B(1, (0+tidx));
		}
		{
			shared_1[(tidx+0)][6]=B(2, (0+tidx));
		}
		{
			shared_1[(tidx+0)][7]=B(3, (0+tidx));
		}
	}
	__syncthreads();
	#pragma unroll
	for (it_2=0; it_2<16; it_2=(it_2+1))
	{
		float a;
		float b_4;
		float b_5;
		float b_6;
		float b_7;
		a=shared_0[((tidx+(( - 1)*(it_2+0)))+16)];
		b_4=shared_1[it_2][4];
		b_5=shared_1[it_2][5];
		b_6=shared_1[it_2][6];
		b_7=shared_1[it_2][7];
		sum_4+=(a*b_4);
		sum_5+=(a*b_5);
		sum_6+=(a*b_6);
		sum_7+=(a*b_7);
	}
	{
		C(((idy*8)+4), idx)=sum_4;
	}
	__syncthreads();
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*8)+(( - 1)*(0-5)))+h), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*8)+(( - 1)*(0-5)))+h), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_1[(tidx+0)][5]=B(0, (0+tidx));
		}
		{
			shared_1[(tidx+0)][6]=B(1, (0+tidx));
		}
		{
			shared_1[(tidx+0)][7]=B(2, (0+tidx));
		}
	}
	__syncthreads();
	#pragma unroll
	for (it_2=0; it_2<16; it_2=(it_2+1))
	{
		float a;
		float b_5;
		float b_6;
		float b_7;
		a=shared_0[((tidx+(( - 1)*(it_2+0)))+16)];
		b_5=shared_1[it_2][5];
		b_6=shared_1[it_2][6];
		b_7=shared_1[it_2][7];
		sum_5+=(a*b_5);
		sum_6+=(a*b_6);
		sum_7+=(a*b_7);
	}
	{
		C(((idy*8)+5), idx)=sum_5;
	}
	__syncthreads();
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*8)+(( - 1)*(0-6)))+h), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*8)+(( - 1)*(0-6)))+h), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_1[(tidx+0)][6]=B(0, (0+tidx));
		}
		{
			shared_1[(tidx+0)][7]=B(1, (0+tidx));
		}
	}
	__syncthreads();
	#pragma unroll
	for (it_2=0; it_2<16; it_2=(it_2+1))
	{
		float a;
		float b_6;
		float b_7;
		a=shared_0[((tidx+(( - 1)*(it_2+0)))+16)];
		b_6=shared_1[it_2][6];
		b_7=shared_1[it_2][7];
		sum_6+=(a*b_6);
		sum_7+=(a*b_7);
	}
	{
		C(((idy*8)+6), idx)=sum_6;
	}
	__syncthreads();
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*8)+(( - 1)*(0-7)))+h), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*8)+(( - 1)*(0-7)))+h), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_1[(tidx+0)][7]=B(0, (0+tidx));
		}
	}
	__syncthreads();
	#pragma unroll
	for (it_2=0; it_2<16; it_2=(it_2+1))
	{
		float a;
		float b_7;
		a=shared_0[((tidx+(( - 1)*(it_2+0)))+16)];
		b_7=shared_1[it_2][7];
		sum_7+=(a*b_7);
	}
	{
		C(((idy*8)+7), idx)=sum_7;
	}
	__syncthreads();
	__syncthreads();
}


#endif // #ifndef _CONV_KERNEL_H_
