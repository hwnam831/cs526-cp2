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
#include <cublas.h>
#include "strsm_gold.cpp"

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

// These are the inline versions for all of the SDK helper functions
void __checkCudaErrors( cudaError err, const char *file, const int line )
{
    if( cudaSuccess != err) {
        fprintf(stderr, "%s(%i) : cudaSafeCall() Runtime API error %d: %s.\n",
                file, line, (int)err, cudaGetErrorString( err ) );
        exit(-1);
    }
}

void __checkCudaErrors( CUresult err, const char *file, const int line )
{
    if( CUDA_SUCCESS != err) {
        fprintf(stderr, "checkCudaErrors() Driver API error = %04d \from file <%s>, line %i.\n",
                err, file, line );

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

// Allocates a matrix with random float entries.
void randomInit(float* data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
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

#define INPUT_WIDTH 8192
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
    float*   h_A   = 0;
    float*   h_B   = 0;
    float*   h_C   = 0;
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
    checkCudaErrors(initCUDA(&hContext, &hDevice, &hModule, &hKernel, ptx, argv[2] ));

    unsigned int num_elements = INPUT_WIDTH;

    // allocate host memory for matrices A and B
    const unsigned int in_mem_size = sizeof( float) * (num_elements*num_elements);
    const unsigned int out_mem_size = sizeof( float) * (num_elements*num_elements);
    if ((h_A = (float*) malloc(in_mem_size)) == NULL) {
        fprintf(stderr, "Could not allocate host memory\n");
        exit(-1);
    }
    if ((h_B = (float*) malloc(in_mem_size)) == NULL) {
        fprintf(stderr, "Could not allocate host memory\n");
        exit(-1);
    }

    // initialize host memory
    for( unsigned int i = 0; i < num_elements; ++i)
    {
        for( unsigned int j = 0; j < num_elements; ++j) {
            h_A[i*num_elements+j] = ((rand()/(float)RAND_MAX));
            if (i>j) h_A[i*num_elements+j]=0.0f;
            h_B[i*num_elements+j] = ((rand()/(float)RAND_MAX));
        }
    }

    // allocate host memory for the result
    if ((h_C = (float*) malloc(out_mem_size)) == NULL) {
        fprintf(stderr, "Could not allocate host memory\n");
        exit(-1);
    }

    checkCudaErrors(cudaMalloc((void**) &d_A, in_mem_size));
    checkCudaErrors(cudaMalloc((void**) &d_C, out_mem_size));

    // copy host memory to device
    checkCudaErrors(cudaMemcpy((void*) d_A, h_A, in_mem_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy((void*) d_C, h_B, in_mem_size, cudaMemcpyHostToDevice));

    float* reference = (float*) malloc(out_mem_size);
    if (reference == NULL) {
        fprintf(stderr, "Could not allocate reference memory\n");
        exit(-1);
    }

    computeGold(h_A, h_B, num_elements, reference);

    // setup execution parameters
    int block_width = 256;

    cublasStrsm('L', 'L', 'N', 'N', num_elements, num_elements, 1.0, (float*)d_A, num_elements, (float*)d_C, num_elements);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaMemcpy(reference, (void*) d_C, out_mem_size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy((void*) d_A, h_A, in_mem_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy((void*) d_C, h_B, in_mem_size, cudaMemcpyHostToDevice));
    
    for (int i=0; i<num_elements; i+=block_width) {
        cublasStrsm('L', 'L', 'N', 'N', block_width, num_elements, 1.0, (float*)d_A+i*num_elements+i, num_elements, (float*)d_C+i, num_elements);
        // left matrix (i,i) (i+64, i+64)        right matrix (0,i) (0, i+64)

        // strsm to get the result matrix (0,i) (0, i+64)
        // result(0, i+64) (0, h) - left matrix (i, i+64) (i+64,h) * result matrix (0,i) (0, i+64)
        dim3 threads(block_width, 1);
        int WC = num_elements - i - block_width;
        if (WC==0) break;
        int HC = num_elements;
        dim3 grid(WC / threads.x, HC / threads.y);

        int i_val = i;
        void *params[] = { &d_C, &d_A, &d_C, &block_width, &num_elements, &i_val };
        // Launch the kernel
        checkCudaErrors(cuLaunchKernel(hKernel, grid.x, grid.y, 1, threads.x, threads.y, 1,
                                       0, NULL, params, NULL));
    }

    cudaDeviceSynchronize();
    fprintf(stderr, "CUDA kernel launched\n");
    // Copy the result back to the host
    checkCudaErrors(cudaMemcpy(h_C, (void*) d_C, out_mem_size, cudaMemcpyDeviceToHost));

    // compute reference solution
    
    int res = checkarray(reference, h_C, num_elements);
    printf("Test %s \n", (res == 0) ? "PASSED" : "FAILED");

    if (res != 0) {
        printDiff(reference, h_C,  num_elements, num_elements);
    }
    
    // Cleanup
    checkCudaErrors(cudaFree((void *) d_A));
    checkCudaErrors(cudaFree((void *) d_B));
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

