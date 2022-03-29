
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cublas.h>

// includes, project
#include <cutil_inline.h>

// includes, kernels
#include <strsm_kernel.cu>
////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int size);

////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C"
void computeGold( float* a, float* b, const unsigned int len, float* result);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char** argv)
{
    CUT_DEVICE_INIT(argc, argv);
	runTest(INPUT_WIDTH);

    CUT_EXIT(argc, argv);
}


int checkarray(float* reference, float* o_data, int num_elements) {
    {
		int error = 0;
		for (int i=0; i<num_elements; i++) {
			for (int j=0; j<num_elements; j++) {
				float t = reference[j*num_elements+i]-o_data[j*num_elements+i];
				if (t<0) t = -t;
				float ref = reference[j*num_elements+i];
				if  (ref<0) ref = -ref;
				if (t/ref>1e-3) {
					if (error<4)
						printf("%d, %d, %f, %f\n", i, j, reference[j*num_elements+i], o_data[j*num_elements+i]);
					error++;
				}
			}
		}
		return error;
    }
}
////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest(int size)
{
    cublasStatus status;
    status = cublasInit();
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! CUBLAS initialization error\n");
	exit (1);
    }

    unsigned int num_elements = size;

    unsigned int timer;
    cutilCheckError( cutCreateTimer(&timer));

    const unsigned int in_mem_size = sizeof( float) * (num_elements*num_elements);
    const unsigned int out_mem_size = sizeof( float) * (num_elements*num_elements);


    // allocate host memory to store the input data
    float* a_data = (float*) malloc( in_mem_size);
    float* b_data = (float*) malloc( in_mem_size);

    float* o_data = (float*) malloc( out_mem_size);
    float* reference = (float*) malloc( out_mem_size);

    // initialize the input data on the host to be integer values
    // between 0 and 1000
    for( unsigned int i = 0; i < num_elements; ++i)
    {
	    for( unsigned int j = 0; j < num_elements; ++j) {
	    	a_data[i*num_elements+j] = ((rand()/(float)RAND_MAX));
	    	if (i>j) a_data[i*num_elements+j]=0.0f;
	    	b_data[i*num_elements+j] = ((rand()/(float)RAND_MAX));
	    }


    }
//    printf("\n");

    // compute reference solution
    computeGold(a_data, b_data, num_elements, reference);

    // allocate device memory input and output arrays
    float* d_a_data;
    float* d_odata;
    cutilSafeCall( cudaMalloc( (void**) &d_a_data, in_mem_size));
    cutilSafeCall( cudaMalloc( (void**) &d_odata, out_mem_size));

    // copy host memory to device input array

    // setup execution parameters
    // Note that these scans only support a single thread-block worth of data,
    // but we invoke them here on many blocks so that we can accurately compare
    // performance

    // make sure there are no CUDA errors before we start
    cutilCheckMsg("Kernel execution failed");

    printf("Running %d elements\n", num_elements);
	float epsilon = 1e-4;

    // execute the kernels
    unsigned int numIterations = 1;




    {
        cutilSafeCall( cudaMemcpy( d_a_data, a_data, in_mem_size, cudaMemcpyHostToDevice) );
        cutilSafeCall( cudaMemcpy( d_odata, b_data, in_mem_size, cudaMemcpyHostToDevice) );


		cudaThreadSynchronize();
		cutStartTimer(timer);

		cublasStrsm('L', 'L', 'N', 'N', num_elements, num_elements, 1.0, d_a_data, num_elements, d_odata, num_elements);
		cudaThreadSynchronize();
		cutStopTimer(timer);
		printf("Average time: %f ms\n", cutGetTimerValue(timer));
		cutResetTimer(timer);



		// copy result from device to host
		cutilSafeCall(cudaMemcpy( reference, d_odata, out_mem_size,
								   cudaMemcpyDeviceToHost));



//		unsigned int result_regtest = cutComparefe( reference, o_data, num_elements*num_elements, epsilon);
//		printf( "cublas: Test %s\n", (1 == result_regtest) ? "PASSED" : "FAILED");
    }

//    // we do the transpose
//    for (int i=0; i<num_elements; i++) {
//        for (int j=0; j<num_elements; j++) {
//        	if (i<j) {
//        		float t = b_data[j*num_elements+i];
//        		b_data[j*num_elements+i] = b_data[i*num_elements+j];
//        		b_data[i*num_elements+j] = t;
//        		t = a_data[j*num_elements+i];
//        		a_data[j*num_elements+i] = a_data[i*num_elements+j];
//        		a_data[i*num_elements+j] = t;
//        	}
//        }
//    }


    {
        cutilSafeCall( cudaMemcpy( d_a_data, a_data, in_mem_size, cudaMemcpyHostToDevice) );
        cutilSafeCall( cudaMemcpy( d_odata, b_data, in_mem_size, cudaMemcpyHostToDevice) );
    	int block_width = 256;
		cudaThreadSynchronize();
		cutStartTimer(timer);

    	for (int i=0; i<num_elements; i+=block_width) {
    		cublasStrsm('L', 'L', 'N', 'N', block_width, num_elements, 1.0, d_a_data+i*num_elements+i, num_elements, d_odata+i, num_elements);
    		// left matrix (i,i) (i+64, i+64)        right matrix (0,i) (0, i+64)


    		// strsm to get the result matrix (0,i) (0, i+64)
    		// result(0, i+64) (0, h) - left matrix (i, i+64) (i+64,h) * result matrix (0,i) (0, i+64)
    		dim3 threads(block_width, 1);
    		int WC = num_elements - i - block_width;
    		if (WC==0) break;
    		int HC = num_elements;
    		dim3 grid(WC / threads.x, HC / threads.y / 16);
    		matmul_opt<<<grid, threads>>>(d_odata+i, d_a_data+(i+block_width)+i*num_elements, d_odata+i+block_width, block_width, num_elements);
    	}
		cudaThreadSynchronize();
		cutStopTimer(timer);
		printf("Average time: %f ms\n", cutGetTimerValue(timer));
		cutResetTimer(timer);


		cutilSafeCall(cudaMemcpy( o_data, d_odata, out_mem_size,
								   cudaMemcpyDeviceToHost));
	    int res = checkarray(reference, o_data, num_elements);
	    printf("Test %s %d\n", (0 == res) ? "PASSED" : "FAILED", res);

	}


    {
        cutilSafeCall( cudaMemcpy( d_a_data, a_data, in_mem_size, cudaMemcpyHostToDevice) );
        cutilSafeCall( cudaMemcpy( d_odata, b_data, in_mem_size, cudaMemcpyHostToDevice) );
    	int block_width = 256;
		cudaThreadSynchronize();
		cutStartTimer(timer);

    	for (int i=0; i<num_elements; i+=block_width) {
    		cublasStrsm('L', 'L', 'N', 'N', block_width, num_elements, 1.0, d_a_data+i*num_elements+i, num_elements, d_odata+i, num_elements);
    		// left matrix (i,i) (i+64, i+64)        right matrix (0,i) (0, i+64)


    		// strsm to get the result matrix (0,i) (0, i+64)
    		// result(0, i+64) (0, h) - left matrix (i, i+64) (i+64,h) * result matrix (0,i) (0, i+64)
    		dim3 threads(32, 1);
    		int WC = num_elements - i - block_width;
    		if (WC==0) break;
    		int HC = num_elements;
    		dim3 grid(WC / threads.x, HC / threads.y / 1);
    		matmul_coalesced<<<grid, threads>>>(d_odata+i, d_a_data+(i+block_width)+i*num_elements, d_odata+i+block_width, block_width, num_elements);
    	}
		cudaThreadSynchronize();
		cutStopTimer(timer);
		printf("matmul_coalesced Average time: %f ms\n", cutGetTimerValue(timer));
		cutResetTimer(timer);


		cutilSafeCall(cudaMemcpy( o_data, d_odata, out_mem_size,
								   cudaMemcpyDeviceToHost));
	    int res = checkarray(reference, o_data, num_elements);
	    printf("Test %s %d\n", (0 == res) ? "PASSED" : "FAILED", res);

	}


    {
        cutilSafeCall( cudaMemcpy( d_a_data, a_data, in_mem_size, cudaMemcpyHostToDevice) );
        cutilSafeCall( cudaMemcpy( d_odata, b_data, in_mem_size, cudaMemcpyHostToDevice) );
    	int block_width = 256;
		cudaThreadSynchronize();
		cutStartTimer(timer);

    	for (int i=0; i<num_elements; i+=block_width) {
    		cublasStrsm('L', 'L', 'N', 'N', block_width, num_elements, 1.0, d_a_data+i*num_elements+i, num_elements, d_odata+i, num_elements);
    		// left matrix (i,i) (i+64, i+64)        right matrix (0,i) (0, i+64)


    		// strsm to get the result matrix (0,i) (0, i+64)
    		// result(0, i+64) (0, h) - left matrix (i, i+64) (i+64,h) * result matrix (0,i) (0, i+64)
    		dim3 threads(block_width, 1);
    		int WC = num_elements - i - block_width;
    		if (WC==0) break;
    		int HC = num_elements;
    		dim3 grid(WC / threads.x, HC / threads.y);
    		matrix_naive<<<grid, threads>>>(d_odata+i, d_a_data+(i+block_width)+i*num_elements, d_odata+i+block_width, block_width, num_elements);
    	}
		cudaThreadSynchronize();
		cutStopTimer(timer);
		printf("naive Average time: %f ms\n", cutGetTimerValue(timer));
		cutResetTimer(timer);


		cutilSafeCall(cudaMemcpy( o_data, d_odata, out_mem_size,
								   cudaMemcpyDeviceToHost));
	    int res = checkarray(reference, o_data, num_elements);
	    printf("Test %s %d\n", (0 == res) ? "PASSED" : "FAILED", res);

	}




    // cleanup memory
    free( a_data);
    free( o_data);
    free( reference);
    cutilSafeCall(cudaFree(d_a_data));
    cutilSafeCall(cudaFree(d_odata));
    cutilCheckError(cutDeleteTimer(timer));
    status = cublasShutdown();
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! shutdown error\n");
    }
    cudaThreadExit();

}


