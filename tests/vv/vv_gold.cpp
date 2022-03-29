

#include <stdio.h>
#include <math.h>
#include <float.h>

////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C"
void computeGold( float* a, float* b, const unsigned int len, float* result);

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! Each element is the sum of the elements before it in the array.
//! @param reference  reference data, computed but preallocated
//! @param idata      input data as provided to device
//! @param len        number of elements in reference / idata
////////////////////////////////////////////////////////////////////////////////
void computeGold( float* a, float* b, const unsigned int len, float* result)
{
	for (unsigned int i=0; i<len; i++) {
		for (unsigned int j=0; j<len; j++) {
			result[j*len+i] += a[j]*b[i];
		}
	}


}
