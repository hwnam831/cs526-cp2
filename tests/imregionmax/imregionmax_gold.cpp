extern "C"
void computeGold(float*,const float*, unsigned int, unsigned int);


#define MaxF(x,y) (x>y?x:y)



void
computeGold(float* r_G, const float* GPU_G, unsigned int wA, unsigned int hA )
{
	for (unsigned int i = 0; i < hA-16; ++i){
        for (unsigned int j = 0; j < wA-16; ++j) {
        	float r0 = GPU_G[(i+15-1)*wA+j+15-1];
        	float r1 = GPU_G[(i+15-1)*wA+j+15];
        	float r2 = GPU_G[(i+15-1)*wA+j+15+1];
        	float r3 = GPU_G[(i+15)*wA+j+15-1];
        	float r4 = GPU_G[(i+15)*wA+j+15];
        	float r5 = GPU_G[(i+15)*wA+j+15+1];
        	float r6 = GPU_G[(i+15+1)*wA+j+15-1];
        	float r7 = GPU_G[(i+15+1)*wA+j+15];
        	float r8 = GPU_G[(i+15+1)*wA+j+15+1];

        	float max = MaxF(r0, r1);
        	max = MaxF(max, r2);
        	max = MaxF(max, r3);
        	max = MaxF(max, r5);
        	max = MaxF(max, r6);
        	max = MaxF(max, r7);
        	max = MaxF(max, r8);

        	r_G[i*(wA-16)+j] = max>r4?0:1;

        }
	}
}
