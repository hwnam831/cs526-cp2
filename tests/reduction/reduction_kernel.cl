
#include "reduction.h"

#define bidx (get_group_id(0))
#define bidy (get_group_id(1))
#define tidx (get_local_id(0))
#define blockDimX (get_local_size(0))
#define gridDimX (get_num_groups(0))
#define idx (bidx*blockDimX+tidx)
/*
__kernel void reduction_old(__global float* d_odata, __global float* d_idata, int num_elements)
{

	d_odata[idx] = d_idata[idx]+d_idata[idx+num_elements/2];
}
*/
__kernel void reduction(__global float* d_odata, __global float* d_idata, int num_elements)
{
	int i;
	float sum = 0.0;
	for (i=0; i<TILE; i++){
		sum += d_idata[idx*TILE+i];
	}
	d_odata[idx] = sum;
}

__kernel void reduction(__global float* d_odata, __global float* d_idata, int num_elements)
{
	int i;
	float sum = 0.0;
	__local float shared[1024];
	for (i=0; i<TILE; i++){
		for (it_2=0; it_2<32; it_2++){
			shared[it_2 * 32 + tidx] = d_idata[(bidx * blockDimX + it_2)* 128 + i + tidx];
		}
		sum += d_idata[idx*TILE+i];
	}
	d_odata[idx] = sum;
}