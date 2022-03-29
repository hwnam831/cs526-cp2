
////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C"
void computeGold( float*, const float*, const float*, unsigned int, unsigned int, unsigned int);


void
computeGold(float* C, const float* A, const float* B, unsigned int hA, unsigned int wA, unsigned int wB)
{
	for (unsigned int i = 0; i < hA; ++i) {
        double sum = 0;
        for (unsigned int j = 0; j < wA; ++j) {
			double a = A[i * wA + j];
			double b = B[j];
			sum += a * b;
        }
		C[i] = (float)sum;
    }
}
