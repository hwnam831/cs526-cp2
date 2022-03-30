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
#include "matrixMul.h"
#include "matrixMul_kernel.cu"

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
                  const char *ptx,
                  const char *kernelname)
{
    // Initialize 
    *phDevice = cudaDeviceInit();

    // Create context on the device
    checkCudaErrors(cuCtxCreate(phContext, 0, *phDevice));

    // Load the PTX 
    checkCudaErrors(cuModuleLoadDataEx(phModule, ptx, 0, 0, 0));

    // Locate the kernel entry poin

    checkCudaErrors(cuModuleGetFunction(phKernel, *phModule, kernelname));


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
computeGold(float* C, const float* A, const float* B, unsigned int hA, unsigned int wA, unsigned int wB)
{
    for (unsigned int i = 0; i < hA; ++i)
        for (unsigned int j = 0; j < wB; ++j) {
            double sum = 0;
            for (unsigned int k = 0; k < wA; ++k) {
                double a = A[i * wA + k];
                double b = B[k * wB + j];
                sum += a * b;
            }
            C[i * wB + j] = (float)sum;
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
    
    const char *filename = argv[1];

    
    char *ll = loadProgramSource(filename, &size);
    fprintf(stdout, "NVVM IR ll file loaded\n");

    // Use libnvvm to generte PTX
    ptx = loadProgramSource(filename, &size);
    fprintf(stdout, "PTX generated:\n");
    fprintf(stdout, "%s\n", ptx);
    
/*
    std::ifstream t(filename);
    if(!t.is_open()) {
        fprintf(stderr, "file not found\n");
        exit(-1);
    }
    std::string str((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
    fprintf(stdout, "%s\n", str.c_str());
*/
    // Initialize the device and get a handle to the kernel
    checkCudaErrors(initCUDA(&hContext, &hDevice, &hModule, &hKernel, ptx, argv[2]));
    
    // set seed for rand()
    srand(2006);

    // allocate host memory for matrices A and B
    unsigned int size_A = WA * HA;
    unsigned int mem_size_A = sizeof(float) * size_A;
    if ((h_A = (float*) malloc(mem_size_A)) == NULL) {
        fprintf(stderr, "Could not allocate host memory\n");
        exit(-1);
    }
    unsigned int size_B = WB * HB;
    unsigned int mem_size_B = sizeof(float) * size_B;
    if ((h_B = (float*) malloc(mem_size_B)) == NULL) {
        fprintf(stderr, "Could not allocate host memory\n");
        exit(-1);
    }

    // initialize host memory
    randomInit(h_A, size_A);
    randomInit(h_B, size_B);

    // allocate device memory for result
    unsigned int size_C = WC * HC;
    unsigned int mem_size_C = sizeof(float) * size_C;

    // allocate host memory for the result
    if ((h_C = (float*) malloc(mem_size_C)) == NULL) {
        fprintf(stderr, "Could not allocate host memory\n");
        exit(-1);
    }

    checkCudaErrors(cuMemAlloc(&d_A, mem_size_A));
    checkCudaErrors(cuMemAlloc(&d_B, mem_size_B));
    checkCudaErrors(cuMemAlloc(&d_C, mem_size_C));

    // copy host memory to device
    checkCudaErrors(cuMemcpyHtoD(d_A, h_A, mem_size_A));
    checkCudaErrors(cuMemcpyHtoD(d_B, h_B, mem_size_B));

    // setup execution parameters
    dim3 threads(16, 16);
    dim3 grid(WC / threads.x, HC / threads.y);

    int Width_A = WA;
    int Width_B = WB;
    void *params[] = { &d_A, &d_B, &d_C, &Width_A, &Width_B };
    // Launch the kernel
    checkCudaErrors(cuLaunchKernel(hKernel, grid.x, grid.y, 1, threads.x, threads.y, 1,
                                   0, NULL, params, NULL)); 

    cudaDeviceSynchronize();
    fprintf(stderr, "CUDA kernel launched\n");
    // Copy the result back to the host
    checkCudaErrors(cuMemcpyDtoH(h_C, d_C, mem_size_C));

    // compute reference solution
    float* reference = (float*) malloc(mem_size_C);
    if (reference == NULL) {
        fprintf(stderr, "Could not allocate reference memory\n");
        exit(-1);
    }
    computeGold(reference, h_A, h_B, HA, WA, WB);

    bool res = cutCompareL2fe(reference, h_C, size_C, 1e-6f);
    printf("Test %s \n", res ? "PASSED" : "FAILED");

    if (!res) {
        //printDiff(reference, h_C,  WC, HC);
    }
    
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
