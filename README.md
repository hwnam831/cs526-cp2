# cs526-cp2

## Requirements
LLVM configured as `cmake -S llvm -B build -G Ninja -DCMAKE_INSTALL_PREFIX="~/llvm" -DLLVM_ENABLE_PROJECTS="clang"`  
`cmake --build build && cmake --install build`    
CUDA installed at `/usr/local/cuda`  
`sudo apt install mesa-common-dev`  
llvm installed or linked to `./llvm`, or set Makefile's LLVMROOT accordingly  
libclc (https://github.com/llvm-mirror/libclc) built (only nvptx) and installed to LLVMROOT  
Example configure script (change --prefix and --with-llvm-config accordingly):  
`./configure.py --prefix=/home/hwnam/llvm --with-llvm-config=/home/hwnam/llvm/bin/llvm-config nvptx--nvidiacl nvptx64--nvidiacl`

## Test Suite
##### The list of algorithms tested in the paper (https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.870.3097&rep=rep1&type=pdf):
transpose matrix vector (tmv)  
martrix multiplication (matmul)  
matrix-vector multiplication (mv)  
vector-vector multiplication (vv)  
reduction (reduction)  
matrix equation solver (strsm)  
convolution (conv)  
matrix transpose (transpose)  
~~reconstruct image (demosaic)~~ <br>
~~find the regional maxima (imregionmax)~~

### To Run Test
`cd tests`  
Run all tests: `make all`  
Clean all output files: `make cleanAll`  
Run individial test (take tmv for example): `cd tmv && make`  
- if Test PASSED is printed, then test passes. Otherwise, if test fails, Test FAILED will be printed.<br>
Clean individial test outputs (take tmv for example): `cd tmv && make clean`
Run test case with the original naive kernel (take tmv for example): `cd tmv && make original`
Profile the execution time for naive, coalesced, and prefetched kernels (take tmv for example): `cd tmv && make profile`
To generate intermeidate ll files after each pass (take tmv for example): `cd tmv && make debug`
- The ll for original naive kernel can be found in tmv.nvvm.ll
- The ll for coalesced kernel can be found in tmv.coal.ll
- The ll for prefetched kernel can be found in tmv.opt.ll
