////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C"
void computeGold( float*, const float*, const float*, unsigned int, unsigned int, unsigned int, unsigned int);

#include "conv.h"

void
computeGold(float* C, const float* A, const float* B, unsigned int hA, unsigned int wA, unsigned int hB, unsigned int wB)
{
	int pwb = (wB<WIDTH_PADDING?WIDTH_PADDING:wB);
    for (unsigned int j = 0; j < hA; ++j) {
    	for (unsigned int i = 0; i < wA; ++i) {
            double sum = 0;
            for (unsigned int j1 = 0; j1 < hB; ++j1) {
                for (unsigned int i1 = 0; i1 < wB; ++i1) {
					sum += A[(j-j1+hB)*(wA+pwb)+i-i1+pwb] * B[j1*wB+i1];
                }
            }
            C[j * wA + i] = (float)sum;
        }
    }
}
