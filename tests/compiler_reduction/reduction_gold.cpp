#include <stdio.h>
#include <math.h>
#include <float.h>

////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C"
void computeGold( float* input, const unsigned int len, float* result);

void
computeGold( float* input, const unsigned int len, float* result)
{
	result[0] = 0;
	for (int i=0; i<len; i++) {
		result[0] += input[i];
	}
}
