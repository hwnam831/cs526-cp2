
#include "reduction.h"

#define bidx (get_group_id(0))
#define bidy (get_group_id(1))
#define tidx (get_local_id(0))
#define blockDimX (get_local_size(0))
#define gridDimX (get_num_groups(0))
#define idx ((bidy*gridDimX+bidx)*blockDimX+tidx)
__kernel void reduction(__global float* d_odata, __global float* d_idata, int num_elements)
{

	d_odata[idx] = d_idata[idx]+d_idata[idx+num_elements/2];
}

