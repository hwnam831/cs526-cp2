/*
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <math.h>
#include <cuda.h>
#include <builtin_types.h>
#include "nvvm.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include "reduction.h"

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

// These are the inline versions for all of the SDK helper functions
void __checkCudaErrors( CUresult err, const char *file, const int line )
{
    if( CUDA_SUCCESS != err) {
        fprintf(stderr, "checkCudaErrors() Driver API error = %04d \from file <%s>, line %i.\n",
                err, file, line );
        //cudaError_t error = cudaGetLastError();
        //fprintf(stderr, "%s\n", cudaGetErrorString(error));
        const char *p;
        cuGetErrorString(err, &p);
        fprintf(stderr, "%s\n", p);
        exit(-1);
    }
}

CUdevice cudaDeviceInit()
{
    CUdevice cuDevice = 0;
    int deviceCount = 0;
    CUresult err = cuInit(0);
    char name[100];
    int major=0, minor=0;

    if (CUDA_SUCCESS == err)
        checkCudaErrors(cuDeviceGetCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "cudaDeviceInit error: no devices supporting CUDA\n");
        exit(-1);
    }
    checkCudaErrors(cuDeviceGet(&cuDevice, 0));
    cuDeviceGetName(name, 100, cuDevice);
    printf("Using CUDA Device [0]: %s\n", name);

    checkCudaErrors( cuDeviceComputeCapability(&major, &minor, cuDevice) );
    if (major < 2) {
        fprintf(stderr, "Device 0 is not sm_20 or later\n");
        exit(-1);
    }
    return cuDevice;
}


CUresult initCUDA(CUcontext *phContext,
                  CUdevice *phDevice,
                  CUmodule *phModule,
                  CUfunction *phKernel,
                  CUfunction *phKernel2,
                  const char *ptx,
                  const char *kernelname, const char *kernelname2)
{
    // Initialize 
    *phDevice = cudaDeviceInit();

    // Create context on the device
    checkCudaErrors(cuCtxCreate(phContext, 0, *phDevice));

    // Load the PTX 
    checkCudaErrors(cuModuleLoadDataEx(phModule, ptx, 0, 0, 0));

    // Locate the kernel entry poin

    checkCudaErrors(cuModuleGetFunction(phKernel, *phModule, kernelname));
    checkCudaErrors(cuModuleGetFunction(phKernel2, *phModule, kernelname2));


    return CUDA_SUCCESS;
}

char *loadProgramSource(const char *filename, size_t *size) 
{
    struct stat statbuf;
    FILE *fh;
    char *source = NULL;
    *size = 0;
    fh = fopen(filename, "rb");
    if (fh) {
        stat(filename, &statbuf);
        source = (char *) malloc(statbuf.st_size+1);
        if (source) {
            fread(source, statbuf.st_size, 1, fh);
            source[statbuf.st_size] = 0;
            *size = statbuf.st_size+1;
        }
    }
    else {
        fprintf(stderr, "Error reading file %s\n", filename);
        exit(-1);
    }
    return source;
}

char *generatePTX(const char *ll, size_t size, const char *filename)
{
    nvvmResult result;
    nvvmProgram program;
    size_t PTXSize;
    char *PTX = NULL;

    result = nvvmCreateProgram(&program);
    if (result != NVVM_SUCCESS) {
        fprintf(stderr, "nvvmCreateProgram: Failed\n");
        exit(-1); 
    }

    result = nvvmAddModuleToProgram(program, ll, size, filename);
    if (result != NVVM_SUCCESS) {
        fprintf(stderr, "nvvmAddModuleToProgram: Failed\n");
        exit(-1);
    }
 
    result = nvvmCompileProgram(program,  0, NULL);
    if (result != NVVM_SUCCESS) {
        char *Msg = NULL;
        size_t LogSize;
        fprintf(stderr, "nvvmCompileProgram: Failed\n");
        nvvmGetProgramLogSize(program, &LogSize);
        Msg = (char*)malloc(LogSize);
        nvvmGetProgramLog(program, Msg);
        fprintf(stderr, "%s\n", Msg);
        free(Msg);
        exit(-1);
    }
    
    result = nvvmGetCompiledResultSize(program, &PTXSize);
    if (result != NVVM_SUCCESS) {
        fprintf(stderr, "nvvmGetCompiledResultSize: Failed\n");
        exit(-1);
    }
    
    PTX = (char*)malloc(PTXSize);
    result = nvvmGetCompiledResult(program, PTX);
    if (result != NVVM_SUCCESS) {
        fprintf(stderr, "nvvmGetCompiledResult: Failed\n");
        free(PTX);
        exit(-1);
    }
    
    result = nvvmDestroyProgram(&program);
    if (result != NVVM_SUCCESS) {
      fprintf(stderr, "nvvmDestroyProgram: Failed\n");
      free(PTX);
      exit(-1);
    }
    
    return PTX;
}

void
computeGold( float* input, const unsigned int len, float* result)
{
    result[0] = 0;
    for (int i=0; i<len; i++) {
        result[0] += input[i];
    }
}

// Allocates a matrix with random float entries.
void randomInit(float* data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

// Compare two float arrays using L2-norm with an epsilon tolerance for equality
// same as cutCompareL2fe in cutil.cpp
bool cutCompareL2fe( const float* reference, const float* data,
                const unsigned int len, const float epsilon ) 
{
    float error = 0;
    float ref = 0;

    for( unsigned int i = 0; i < len; ++i) {
        float diff = reference[i] - data[i];
        error += diff * diff;
        ref += reference[i] * reference[i];
    }

    float normRef = sqrtf(ref);
    if (fabs(ref) < 1e-7) {
        fprintf(stderr, "ERROR, reference l2-norm is 0\n");
        return false;
    }
    float normError = sqrtf(error);
    error = normError / normRef;

    return error < epsilon; // l2-norm error is greater than epsilon?
}

void printDiff(float *data1, float *data2, int width, int height)
{
  int i,j,k;
  int error_count=0;
  for (j=0; j<height; j++) {
    for (i=0; i<width; i++) {
      k = j*width+i;
      if (data1[k] != data2[k]) {
         printf("diff(%d,%d) CPU=%4.4f, GPU=%4.4f n", i,j, data1[k], data2[k]);
         error_count++;
      }
    }
  }
  printf(" nTotal Errors = %d n", error_count);
}

int main(int argc, char **argv)
{
    const unsigned int nThreads = 32;
    const unsigned int nBlocks  = 1;
    const size_t memSize = nThreads * nBlocks * sizeof(int);

    CUcontext    hContext = 0;
    CUdevice     hDevice  = 0;
    CUmodule     hModule  = 0;
    CUfunction   hKernel  = 0;
    CUfunction   hKernel2  = 0;
    CUdeviceptr  d_A   = 0;
    CUdeviceptr  d_B   = 0;
    CUdeviceptr  d_C   = 0;
    float         *h_A   = 0;
    float         *h_B   = 0;
    float         *h_C   = 0;
    char        *ptx      = NULL;
    unsigned int i;

    // Get the ll from file
    size_t size = 0;
    // Kernel parameters
    if (argc < 3){
        fprintf(stdout, "Usage: ./loader [PTXFILE] [KERNELNAME]");
        return -1;
    }
    
//TODO
    const char *filename = argv[1];

    /*
    char *ll = loadProgramSource(filename, &size);
    fprintf(stdout, "NVVM IR ll file loaded\n");

    // Use libnvvm to generte PTX
    ptx = loadProgramSource(filename, &size);
    fprintf(stdout, "PTX generated:\n");
    fprintf(stdout, "%s\n", ptx);
    */

    std::ifstream t(filename);
    if(!t.is_open()) {
        fprintf(stderr, "file not found\n");
        exit(-1);
    }
    std::string str((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
    fprintf(stdout, "%s\n", str.c_str());

    // Initialize the device and get a handle to the kernel
    checkCudaErrors(initCUDA(&hContext, &hDevice, &hModule, &hKernel, &hKernel2, str.c_str(), "_Z15reduction_opt_0Pfii", "_Z15reduction_opt_1Pfii"));
    //checkCudaErrors(initCUDA(&hContext, &hDevice, &hModule, &hKernel2, str.c_str(), "_Z15reduction_opt_1Pfii"));

    unsigned int num_elements = INPUT_SIZE;
    unsigned int num_elements_B = 65536;

    // allocate host memory
    const unsigned int mem_size = sizeof( float) * (num_elements);
    const unsigned int output_mem_size = sizeof( float) * (num_elements);
    if ((h_A = (float*) malloc(mem_size)) == NULL) {
        fprintf(stderr, "Could not allocate host memory\n");
        exit(-1);
    }

    // initialize host memory
    //randomInit(h_A, num_elements);
    for( unsigned int i = 0; i < num_elements; ++i)
    {
        h_A[i] = ((rand()/(float)RAND_MAX));
    }

    // allocate host memory for the result
    if ((h_C = (float*) malloc(output_mem_size)) == NULL) {
        fprintf(stderr, "Could not allocate host memory\n");
        exit(-1);
    }

    // compute reference solution
    float* reference = (float*) malloc(output_mem_size);
    if (reference == NULL) {
        fprintf(stderr, "Could not allocate reference memory\n");
        exit(-1);
    }
    computeGold( h_A, num_elements, reference);
    printf( "cpu: Test %f\n", reference[0]);

    checkCudaErrors(cuMemAlloc(&d_A, mem_size));
    checkCudaErrors(cuMemAlloc(&d_B, num_elements_B*sizeof( float)));
    checkCudaErrors(cuMemAlloc(&d_C, output_mem_size));

    // copy host memory to device
    checkCudaErrors(cuMemcpyHtoD(d_A, h_A, mem_size));

    cudaDeviceSynchronize();
    // execute the kernel
    float result = 0.0f;
	unsigned int numIterations = 1;
	for (int i=0; i<numIterations; i++) {
	    dim3  grid(65536/512, 1, 1);
	    dim3  threads(512, 1, 1);
        int num1 = 262144;
        void *params[] = { &d_A, &num_elements, &num1 };
        checkCudaErrors(cuLaunchKernel(hKernel, grid.x, grid.y, grid.z, threads.x, threads.y, threads.z, 0, NULL, params, NULL)); 
	    grid.x = 1;
	threads.x = 512;
        int num2 = 262144;
        void *params2[] = { &d_A, &num_elements, &num2 };
        checkCudaErrors(cuLaunchKernel(hKernel2, grid.x, grid.y, grid.z, threads.x, threads.y, threads.z, 0, NULL, params2, NULL)); 
		
        checkCudaErrors(cuMemcpyDtoH(h_C, d_A, sizeof(float)*1));
		result += h_C[0];
	}

    cudaDeviceSynchronize();
    fprintf(stderr, "CUDA kernel launched\n");
    // Copy the result back to the host

    
    printf("result: %f, %f\n", result, reference[0]);
/*
    bool res = cutCompareL2fe(reference, h_C, 1, 1e-6f);
    printf("Test %s \n", res ? "PASSED" : "FAILED");

    if (!res) {
        printDiff(h_A, h_C, 5, 1);
    }
*/    
    // Cleanup
    checkCudaErrors(cuMemFree(d_A));
    checkCudaErrors(cuMemFree(d_B));
    free(h_A);
    free(h_B);
    free(h_C);
    free(reference);
    if (hModule) {
        checkCudaErrors(cuModuleUnload(hModule));
        hModule = 0;
    }
    if (hContext) {
        checkCudaErrors(cuCtxDestroy(hContext));
        hContext = 0;
    }

    //free(ll);
    //free(ptx);
    
    return 0;
}

