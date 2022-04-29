#define INPUT_WIDTH 8192

#define WIDTH_A INPUT_WIDTH
#define WIDTH_B INPUT_WIDTH
#define WIDTH_C INPUT_WIDTH
#define A(y,x) A[(y)*WIDTH_A+(x)]
#define B(y,x) B[(y)*WIDTH_B+(x)]
#define C(y,x) C[(y)*WIDTH_C+(x)]
__kernel void strsm(__global float *A, __global float *B, __global float *C, int width, int input_width, int i_val)
{
	int tidx = get_local_id(0);
	int tidy = get_local_id(1);
	int idx = get_group_id(0) * get_local_size(0)+tidx;
	int idy = get_group_id(1) * get_local_size(1)+tidy;
    A += i_val;
    B += (i_val+width)+i_val*input_width;
    C += i_val+width;

    float sum = 0;
	#pragma unroll 2
	for (int i=0; i<width; i++) {
		sum+=A(idy, i)*B(i, idx);
	}
	C(idy, idx) -= sum;
}


