

////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C"
void computeGold( float*, const float*, unsigned int, unsigned int, unsigned int);


void
computeGold(float* C, const float* A, unsigned int hA, unsigned int wA, unsigned int wB)
{
    for (unsigned int j = 0; j < wA; ++j) {
		for (unsigned int i = 0; i < hA; ++i) {
			C[j*hA+i]=A[i*wA+j];
        }
    }
}
