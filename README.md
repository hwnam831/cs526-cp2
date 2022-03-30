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
transpose matrix vector (tmv)  
martrix multiplication (matmul)  
matrix-vector multiplication (mv)  
vector-vector multiplication (vv)  
reduction (reduction)  
matrix equation solver (strsm)  
convolution (conv/conv_16_16)  
matrix transpose (transpose)  
reconstruct image (demosaic)  
find the regional maxima (imregionmax)  

`cd tests`  
Run all tests: `make all`  
Clean all files: `make cleanAll`  
Run individial test (take tmv for example): `cd tmv && make`  
Clean individial test (take tmv for example): `cd tmv && make clean`  
