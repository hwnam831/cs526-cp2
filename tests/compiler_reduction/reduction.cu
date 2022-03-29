
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cublas.h>

// includes, project
#include <cutil_inline.h>

// includes, kernels
#include <reduction_kernel.cu>
////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest();

////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C"
void computeGold( float* input, const unsigned int len, float* result);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char** argv)
{
    CUT_DEVICE_INIT(argc, argv);

    runTest();

    CUT_EXIT(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest()
{
    cublasStatus status;
    status = cublasInit();
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! CUBLAS initialization error\n");
	exit (1);
    }

    unsigned int num_elements = INPUT_SIZE;
    unsigned int num_elements_B = 65536;

    unsigned int timer;
    cutilCheckError( cutCreateTimer(&timer));

    const unsigned int mem_size = sizeof( float) * (num_elements);
    const unsigned int output_mem_size = sizeof( float) * (num_elements);


    // allocate host memory to store the input data
    float* h_data = (float*) malloc( mem_size);
    float* o_data = (float*) malloc(output_mem_size);
    float* reference = (float*) malloc(output_mem_size);
    float* b_data = (float*) malloc(num_elements_B*sizeof( float));


    // initialize the input data on the host to be integer values
    // between 0 and 1000
    for( unsigned int i = 0; i < num_elements; ++i)
    {
		h_data[i] = ((rand()/(float)RAND_MAX));
    }
//    printf("\n");

    // compute reference solution
    computeGold( h_data, num_elements, reference);
	printf( "cpu: Test %f\n", reference[0]);

    // allocate device memory input and output arrays
    float* d_idata;
    float* d_odata;
    float* b_idata;
    cutilSafeCall( cudaMalloc( (void**) &d_idata, mem_size));
    cutilSafeCall( cudaMalloc( (void**) &d_odata, output_mem_size));
    cutilSafeCall( cudaMalloc( (void**) &b_idata, num_elements_B*sizeof( float)));


    // copy host memory to device input array
    cutilSafeCall( cudaMemcpy( d_idata, h_data, mem_size, cudaMemcpyHostToDevice) );

    // setup execution parameters
    // Note that these scans only support a single thread-block worth of data,
    // but we invoke them here on many blocks so that we can accurately compare
    // performance

    // make sure there are no CUDA errors before we start
    cutilCheckMsg("Kernel execution failed");

    printf("Running %d elements\n", num_elements);

    // execute the kernels
    unsigned int numIterations = 1;

    float results[1024];

    int pid = 0;



    {
        cutilSafeCall( cudaMemcpy( d_idata, h_data, mem_size, cudaMemcpyHostToDevice) );

		cudaThreadSynchronize();
		cutStartTimer(timer);
		float result = 0.0f;

		for (int i=0; i<numIterations; i++) {
			result += cublasSasum(num_elements, d_idata, 1);
		}
		cudaThreadSynchronize();
		cutStopTimer(timer);
		printf("cublas: Average time: %f ms, %f\n", cutGetTimerValue(timer) / numIterations, result);
		results[pid++] = cutGetTimerValue(timer) / numIterations;
		cutResetTimer(timer);
    }

    {
        cutilSafeCall( cudaMemcpy( d_idata, h_data, mem_size, cudaMemcpyHostToDevice) );

		cudaThreadSynchronize();
		cutStartTimer(timer);
		float result = 0.0f;

		for (int i=0; i<numIterations; i++) {
			result += cublasScasum(num_elements/2, (cuComplex*)d_idata, 1);
		}
		cudaThreadSynchronize();
		cutStopTimer(timer);
		printf("cublas complex: Average time: %f ms, %f\n", cutGetTimerValue(timer) / numIterations, result);
		results[pid++] = cutGetTimerValue(timer) / numIterations;
		cutResetTimer(timer);

    }

    {
        cutilSafeCall( cudaMemcpy( d_idata, h_data, mem_size, cudaMemcpyHostToDevice) );

		cudaThreadSynchronize();
		cutStartTimer(timer);
		float result = 0.0f;


		int flip = 0;
		for (int i=1; i<num_elements; i*=2) {
	        dim3  grid(num_elements/i/2/256, 1, 1);
	        if (grid.x>1024) {
	        	grid.y = grid.x/1024;
	        	grid.x = 1024;
	        }
	        dim3  threads(256, 1, 1);
	        if (grid.x==0) {
	        	grid.x = 1;
	        	threads.x = num_elements/i/2;
	        }
			reduction_naive<<< grid, threads>>>(flip?d_idata:d_odata, flip?d_odata:d_idata, num_elements/i);
			flip = 1-flip;
		}
		cutilSafeCall(cudaMemcpy( o_data, flip?d_odata:d_idata, sizeof(float)*1,
							   cudaMemcpyDeviceToHost));
		result = o_data[0]*numIterations;
		cudaThreadSynchronize();
		cutStopTimer(timer);
		printf("naive: Average time: %f ms, %f\n", cutGetTimerValue(timer) , result);
		results[pid++] = cutGetTimerValue(timer);
		cutResetTimer(timer);

    }




    {
        cutilSafeCall( cudaMemcpy( d_idata, h_data, mem_size, cudaMemcpyHostToDevice) );

		cudaThreadSynchronize();
		cutStartTimer(timer);

		float result = 0.0f;
		numIterations = 1;
		for (int i=0; i<numIterations; i++) {
	        dim3  grid(65536/512, 1, 1);
	        dim3  threads(512, 1, 1);
			reduction_complex_opt_0<<< grid, threads>>>(d_idata, b_idata, num_elements/2, 262144);
			grid.x = 1;
	//		threads.x = 512;
			reduction_complex_opt_1<<< grid, threads>>>(d_idata, b_idata, num_elements/2, 262144);



			cutilSafeCall(cudaMemcpy( o_data, b_idata, sizeof(float)*1,
									   cudaMemcpyDeviceToHost));
			result += o_data[0];
		}
		cudaThreadSynchronize();
		cutStopTimer(timer);
		printf("reduction_complex_opt : Average time: %f ms, %f\n", cutGetTimerValue(timer) / numIterations, result);
		results[pid++] = cutGetTimerValue(timer)/numIterations;
		cutResetTimer(timer);
    }

    {
        cutilSafeCall( cudaMemcpy( d_idata, h_data, mem_size, cudaMemcpyHostToDevice) );

		cudaThreadSynchronize();
		cutStartTimer(timer);

		float result = 0.0f;
		numIterations = 1;
		for (int i=0; i<numIterations; i++) {
	        dim3  grid(65536/512, 1, 1);
	        dim3  threads(512, 1, 1);
			reduction_opt_0<<< grid, threads>>>(d_idata, num_elements, 262144);
			grid.x = 1;
	//		threads.x = 512;
			reduction_opt_1<<< grid, threads>>>(d_idata, num_elements, 262144);



			cutilSafeCall(cudaMemcpy( o_data, d_idata, sizeof(float)*1,
									   cudaMemcpyDeviceToHost));
			result += o_data[0];
		}
		cudaThreadSynchronize();
		cutStopTimer(timer);
		printf("reduction_opt : Average time: %f ms, %f\n", cutGetTimerValue(timer) / numIterations, result);
		results[pid++] = cutGetTimerValue(timer)/numIterations;
		cutResetTimer(timer);
    }


    // cleanup memory
    free( h_data);
    free( o_data);
    free( reference);
    cutilSafeCall(cudaFree(d_idata));
    cutilSafeCall(cudaFree(d_odata));
    cutilCheckError(cutDeleteTimer(timer));
    status = cublasShutdown();
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! shutdown error\n");
    }

    cudaThreadExit();

}


