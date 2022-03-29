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
#define C(y,x) C[(y)*WIDTH_C+(x)]
#define A(y,x) A[(y)*WIDTH_A+(x)]
__global__ void imregionmax_coalesced(float * A, float * C, int width)
{
	__shared__ float shared_0[48];
	float temp[9];
	int t;
	int i;
	t=0;
	#pragma unroll
	for (i=0; i<3; i=(i+1))
	{
		{
			int it_1;
			shared_0[(tidx+0)]=A(((idy+(-1*i))+16), (idx+(-1*0)));
			shared_0[(tidx+16)]=A(((idy+(-1*i))+16), ((idx+(-1*0))+16));
			__syncthreads();
			#pragma unroll
			for (it_1=0; it_1<3; it_1=(it_1+1))
			{
				float a;
				a=shared_0[((tidx+(-1*it_1))+16)];
				temp[t]=a;
				t=(t+1);
			}
			__syncthreads();
		}
	}
	{
		C(idy, idx)=cal(temp);
	}
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
#define merger_y 32
#define coalesced_idy (bidy/(COALESCED_NUM/(merger_y*blockDimY))*COALESCED_NUM)
#define C(y,x) C[(y)*WIDTH_C+(x)]
#define A(y,x) A[(y)*WIDTH_A+(x)]
__global__ void imregionmax_opt(float * A, float * C, int width)
{
	__shared__ float shared_0[272];
	float temp_0[9];
	float temp_1[9];
	float temp_2[9];
	float temp_3[9];
	float temp_4[9];
	float temp_5[9];
	float temp_6[9];
	float temp_7[9];
	float temp_8[9];
	float temp_9[9];
	float temp_10[9];
	float temp_11[9];
	float temp_12[9];
	float temp_13[9];
	float temp_14[9];
	float temp_15[9];
	float temp_16[9];
	float temp_17[9];
	float temp_18[9];
	float temp_19[9];
	float temp_20[9];
	float temp_21[9];
	float temp_22[9];
	float temp_23[9];
	float temp_24[9];
	float temp_25[9];
	float temp_26[9];
	float temp_27[9];
	float temp_28[9];
	float temp_29[9];
	float temp_30[9];
	float temp_31[9];
	int t_0;
	int t_1;
	int t_2;
	int t_3;
	int t_4;
	int t_5;
	int t_6;
	int t_7;
	int t_8;
	int t_9;
	int t_10;
	int t_11;
	int t_12;
	int t_13;
	int t_14;
	int t_15;
	int t_16;
	int t_17;
	int t_18;
	int t_19;
	int t_20;
	int t_21;
	int t_22;
	int t_23;
	int t_24;
	int t_25;
	int t_26;
	int t_27;
	int t_28;
	int t_29;
	int t_30;
	int t_31;
	int it_1;
	t_0=0;
	t_1=0;
	t_2=0;
	t_3=0;
	t_4=0;
	t_5=0;
	t_6=0;
	t_7=0;
	t_8=0;
	t_9=0;
	t_10=0;
	t_11=0;
	t_12=0;
	t_13=0;
	t_14=0;
	t_15=0;
	t_16=0;
	t_17=0;
	t_18=0;
	t_19=0;
	t_20=0;
	t_21=0;
	t_22=0;
	t_23=0;
	t_24=0;
	t_25=0;
	t_26=0;
	t_27=0;
	t_28=0;
	t_29=0;
	t_30=0;
	t_31=0;
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*32)+(( - 1)*(3-1)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*32)+(( - 1)*(3-1)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_0[t_0]=a;
		t_0=(t_0+1);
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*32)+(( - 1)*(3-2)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*32)+(( - 1)*(3-2)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_0[t_0]=a;
		temp_1[t_1]=a;
		t_0=(t_0+1);
		t_1=(t_1+1);
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*32)+(( - 1)*(3-3)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*32)+(( - 1)*(3-3)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_0[t_0]=a;
		temp_1[t_1]=a;
		temp_2[t_2]=a;
		t_0=(t_0+1);
		t_1=(t_1+1);
		t_2=(t_2+1);
	}
	{
		C(((idy*32)+0), idx)=cal(temp_0);
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*32)+(( - 1)*(0-1)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*32)+(( - 1)*(0-1)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_1[t_1]=a;
		temp_2[t_2]=a;
		temp_3[t_3]=a;
		t_1=(t_1+1);
		t_2=(t_2+1);
		t_3=(t_3+1);
	}
	{
		C(((idy*32)+1), idx)=cal(temp_1);
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*32)+(( - 1)*(0-2)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*32)+(( - 1)*(0-2)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_2[t_2]=a;
		temp_3[t_3]=a;
		temp_4[t_4]=a;
		t_2=(t_2+1);
		t_3=(t_3+1);
		t_4=(t_4+1);
	}
	{
		C(((idy*32)+2), idx)=cal(temp_2);
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*32)+(( - 1)*(0-3)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*32)+(( - 1)*(0-3)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_3[t_3]=a;
		temp_4[t_4]=a;
		temp_5[t_5]=a;
		t_3=(t_3+1);
		t_4=(t_4+1);
		t_5=(t_5+1);
	}
	{
		C(((idy*32)+3), idx)=cal(temp_3);
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*32)+(( - 1)*(0-4)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*32)+(( - 1)*(0-4)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_4[t_4]=a;
		temp_5[t_5]=a;
		temp_6[t_6]=a;
		t_4=(t_4+1);
		t_5=(t_5+1);
		t_6=(t_6+1);
	}
	{
		C(((idy*32)+4), idx)=cal(temp_4);
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*32)+(( - 1)*(0-5)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*32)+(( - 1)*(0-5)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_5[t_5]=a;
		temp_6[t_6]=a;
		temp_7[t_7]=a;
		t_5=(t_5+1);
		t_6=(t_6+1);
		t_7=(t_7+1);
	}
	{
		C(((idy*32)+5), idx)=cal(temp_5);
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*32)+(( - 1)*(0-6)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*32)+(( - 1)*(0-6)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_6[t_6]=a;
		temp_7[t_7]=a;
		temp_8[t_8]=a;
		t_6=(t_6+1);
		t_7=(t_7+1);
		t_8=(t_8+1);
	}
	{
		C(((idy*32)+6), idx)=cal(temp_6);
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*32)+(( - 1)*(0-7)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*32)+(( - 1)*(0-7)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_7[t_7]=a;
		temp_8[t_8]=a;
		temp_9[t_9]=a;
		t_7=(t_7+1);
		t_8=(t_8+1);
		t_9=(t_9+1);
	}
	{
		C(((idy*32)+7), idx)=cal(temp_7);
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*32)+(( - 1)*(0-8)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*32)+(( - 1)*(0-8)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_8[t_8]=a;
		temp_9[t_9]=a;
		temp_10[t_10]=a;
		t_8=(t_8+1);
		t_9=(t_9+1);
		t_10=(t_10+1);
	}
	{
		C(((idy*32)+8), idx)=cal(temp_8);
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*32)+(( - 1)*(0-9)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*32)+(( - 1)*(0-9)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_9[t_9]=a;
		temp_10[t_10]=a;
		temp_11[t_11]=a;
		t_9=(t_9+1);
		t_10=(t_10+1);
		t_11=(t_11+1);
	}
	{
		C(((idy*32)+9), idx)=cal(temp_9);
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*32)+(( - 1)*(0-10)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*32)+(( - 1)*(0-10)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_10[t_10]=a;
		temp_11[t_11]=a;
		temp_12[t_12]=a;
		t_10=(t_10+1);
		t_11=(t_11+1);
		t_12=(t_12+1);
	}
	{
		C(((idy*32)+10), idx)=cal(temp_10);
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*32)+(( - 1)*(0-11)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*32)+(( - 1)*(0-11)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_11[t_11]=a;
		temp_12[t_12]=a;
		temp_13[t_13]=a;
		t_11=(t_11+1);
		t_12=(t_12+1);
		t_13=(t_13+1);
	}
	{
		C(((idy*32)+11), idx)=cal(temp_11);
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*32)+(( - 1)*(0-12)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*32)+(( - 1)*(0-12)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_12[t_12]=a;
		temp_13[t_13]=a;
		temp_14[t_14]=a;
		t_12=(t_12+1);
		t_13=(t_13+1);
		t_14=(t_14+1);
	}
	{
		C(((idy*32)+12), idx)=cal(temp_12);
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*32)+(( - 1)*(0-13)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*32)+(( - 1)*(0-13)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_13[t_13]=a;
		temp_14[t_14]=a;
		temp_15[t_15]=a;
		t_13=(t_13+1);
		t_14=(t_14+1);
		t_15=(t_15+1);
	}
	{
		C(((idy*32)+13), idx)=cal(temp_13);
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*32)+(( - 1)*(0-14)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*32)+(( - 1)*(0-14)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_14[t_14]=a;
		temp_15[t_15]=a;
		temp_16[t_16]=a;
		t_14=(t_14+1);
		t_15=(t_15+1);
		t_16=(t_16+1);
	}
	{
		C(((idy*32)+14), idx)=cal(temp_14);
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*32)+(( - 1)*(0-15)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*32)+(( - 1)*(0-15)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_15[t_15]=a;
		temp_16[t_16]=a;
		temp_17[t_17]=a;
		t_15=(t_15+1);
		t_16=(t_16+1);
		t_17=(t_17+1);
	}
	{
		C(((idy*32)+15), idx)=cal(temp_15);
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*32)+(( - 1)*(0-16)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*32)+(( - 1)*(0-16)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_16[t_16]=a;
		temp_17[t_17]=a;
		temp_18[t_18]=a;
		t_16=(t_16+1);
		t_17=(t_17+1);
		t_18=(t_18+1);
	}
	{
		C(((idy*32)+16), idx)=cal(temp_16);
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*32)+(( - 1)*(0-17)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*32)+(( - 1)*(0-17)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_17[t_17]=a;
		temp_18[t_18]=a;
		temp_19[t_19]=a;
		t_17=(t_17+1);
		t_18=(t_18+1);
		t_19=(t_19+1);
	}
	{
		C(((idy*32)+17), idx)=cal(temp_17);
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*32)+(( - 1)*(0-18)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*32)+(( - 1)*(0-18)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_18[t_18]=a;
		temp_19[t_19]=a;
		temp_20[t_20]=a;
		t_18=(t_18+1);
		t_19=(t_19+1);
		t_20=(t_20+1);
	}
	{
		C(((idy*32)+18), idx)=cal(temp_18);
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*32)+(( - 1)*(0-19)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*32)+(( - 1)*(0-19)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_19[t_19]=a;
		temp_20[t_20]=a;
		temp_21[t_21]=a;
		t_19=(t_19+1);
		t_20=(t_20+1);
		t_21=(t_21+1);
	}
	{
		C(((idy*32)+19), idx)=cal(temp_19);
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*32)+(( - 1)*(0-20)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*32)+(( - 1)*(0-20)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_20[t_20]=a;
		temp_21[t_21]=a;
		temp_22[t_22]=a;
		t_20=(t_20+1);
		t_21=(t_21+1);
		t_22=(t_22+1);
	}
	{
		C(((idy*32)+20), idx)=cal(temp_20);
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*32)+(( - 1)*(0-21)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*32)+(( - 1)*(0-21)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_21[t_21]=a;
		temp_22[t_22]=a;
		temp_23[t_23]=a;
		t_21=(t_21+1);
		t_22=(t_22+1);
		t_23=(t_23+1);
	}
	{
		C(((idy*32)+21), idx)=cal(temp_21);
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*32)+(( - 1)*(0-22)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*32)+(( - 1)*(0-22)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_22[t_22]=a;
		temp_23[t_23]=a;
		temp_24[t_24]=a;
		t_22=(t_22+1);
		t_23=(t_23+1);
		t_24=(t_24+1);
	}
	{
		C(((idy*32)+22), idx)=cal(temp_22);
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*32)+(( - 1)*(0-23)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*32)+(( - 1)*(0-23)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_23[t_23]=a;
		temp_24[t_24]=a;
		temp_25[t_25]=a;
		t_23=(t_23+1);
		t_24=(t_24+1);
		t_25=(t_25+1);
	}
	{
		C(((idy*32)+23), idx)=cal(temp_23);
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*32)+(( - 1)*(0-24)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*32)+(( - 1)*(0-24)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_24[t_24]=a;
		temp_25[t_25]=a;
		temp_26[t_26]=a;
		t_24=(t_24+1);
		t_25=(t_25+1);
		t_26=(t_26+1);
	}
	{
		C(((idy*32)+24), idx)=cal(temp_24);
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*32)+(( - 1)*(0-25)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*32)+(( - 1)*(0-25)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_25[t_25]=a;
		temp_26[t_26]=a;
		temp_27[t_27]=a;
		t_25=(t_25+1);
		t_26=(t_26+1);
		t_27=(t_27+1);
	}
	{
		C(((idy*32)+25), idx)=cal(temp_25);
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*32)+(( - 1)*(0-26)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*32)+(( - 1)*(0-26)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_26[t_26]=a;
		temp_27[t_27]=a;
		temp_28[t_28]=a;
		t_26=(t_26+1);
		t_27=(t_27+1);
		t_28=(t_28+1);
	}
	{
		C(((idy*32)+26), idx)=cal(temp_26);
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*32)+(( - 1)*(0-27)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*32)+(( - 1)*(0-27)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_27[t_27]=a;
		temp_28[t_28]=a;
		temp_29[t_29]=a;
		t_27=(t_27+1);
		t_28=(t_28+1);
		t_29=(t_29+1);
	}
	{
		C(((idy*32)+27), idx)=cal(temp_27);
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*32)+(( - 1)*(0-28)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*32)+(( - 1)*(0-28)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_28[t_28]=a;
		temp_29[t_29]=a;
		temp_30[t_30]=a;
		t_28=(t_28+1);
		t_29=(t_29+1);
		t_30=(t_30+1);
	}
	{
		C(((idy*32)+28), idx)=cal(temp_28);
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*32)+(( - 1)*(0-29)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*32)+(( - 1)*(0-29)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_29[t_29]=a;
		temp_30[t_30]=a;
		temp_31[t_31]=a;
		t_29=(t_29+1);
		t_30=(t_30+1);
		t_31=(t_31+1);
	}
	{
		C(((idy*32)+29), idx)=cal(temp_29);
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*32)+(( - 1)*(0-30)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*32)+(( - 1)*(0-30)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_30[t_30]=a;
		temp_31[t_31]=a;
		t_30=(t_30+1);
		t_31=(t_31+1);
	}
	{
		C(((idy*32)+30), idx)=cal(temp_30);
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*32)+(( - 1)*(0-31)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*32)+(( - 1)*(0-31)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_31[t_31]=a;
		t_31=(t_31+1);
	}
	{
		C(((idy*32)+31), idx)=cal(temp_31);
	}
	__syncthreads();
}


#endif // #ifndef _IMREGIONMAX_KERNEL_H_
