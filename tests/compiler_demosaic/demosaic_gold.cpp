extern "C"
void computeGold(float*,const float*, unsigned int, unsigned int);


void
computeGold(float* r_G, const float* GPU_G, unsigned int wA, unsigned int hA )
{
	for (unsigned int i = 0; i < hA-16; ++i){
        for (unsigned int j = 0; j < wA-16; ++j) {
           r_G[i*(wA-16)+j] = GPU_G[(i-1+15)*wA+j+15]*0.25 + GPU_G[(i+15)*wA+j-1+15]*0.25 + GPU_G[(i+15)*wA+j+15] + GPU_G[(i+15)*wA+j+1+15]*0.25 + GPU_G[(i+1+15)*wA+j+15]*0.25;
        }
	}
}
