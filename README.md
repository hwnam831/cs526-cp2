# cs526-cp2
## Environment
Quadro P2200 and Intel(R) Xeon(R) W-2245 CPU with CUDA 11.6 and Ubuntu 21.10

## Requirements
LLVM configured as `cmake -S llvm -B build -G Ninja -DCMAKE_INSTALL_PREFIX="~/llvm" -DLLVM_ENABLE_PROJECTS="clang"`  
`cmake --build build && cmake --install build`    
CUDA installed at `/usr/local/cuda`  
`sudo apt install mesa-common-dev`  
llvm installed or linked to `./llvm`, or set Makefile's LLVMROOT accordingly  
libclc (https://github.com/llvm-mirror/libclc) built (only nvptx) and installed to LLVMROOT  
Example configure script (change --prefix and --with-llvm-config accordingly):  
`./configure.py --prefix=/home/hwnam/llvm --with-llvm-config=/home/hwnam/llvm/bin/llvm-config nvptx--nvidiacl nvptx64--nvidiacl`

## Step-by-Step guide to setup environment
1. Download LLVM 12.0.0 release (llvm-project-12.0.0.src.tar.xz) from https://github.com/llvm/llvm-project/releases/tag/llvmorg-12.0.0
2. Unzip LLVM folder and let's call the unzipped folder llvm
3. Configure by `cmake -S llvm -B build -G Ninja -DCMAKE_INSTALL_PREFIX="~/llvm" -DLLVM_ENABLE_PROJECTS="clang"` (can change the prefix to anywhere you want your LLVMROOT in the ./tests/Makefile to be).
4. Inside llvm, build by `cmake --build build && cmake --install build`
5. Install the CUDA 11.6 Toolkit from https://developer.nvidia.com/cuda-downloads using the corresponding runfile and following the command there.
6. Say CUDA is installed at `/usr/local/cuda`, then this is your CUDAROOT in ./tests/Makefile
7. `sudo apt install mesa-common-dev`
8. Git clone our code from https://github.com/hwnam831/cs526-cp2.git
9. Make sure LLVMROOT and CUDAROOT is updated in the cs526-cp2/tests/Makefile.
10. Git clone libclc from https://github.com/llvm-mirror/libclc.git
11. Inside libclc, run `./configure.py --prefix=~/llvm --with-llvm-config=~/llvm/bin/llvm-config nvptx--nvidiacl nvptx64--nvidiacl`
12. Then make && make install
13. Inside cs526-cp2, run ./cmake.sh
14. Go ./tests
15. Follow the "To Run Test Guide" below to run test.
* You could have the following directory structure: <br>
  ├── cs526-cp2 <br>
  ├── llvm <br>
  ├── libclc <br>

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

### To Run Test Guide
`cd tests`  
* Run all tests: `make all`  
* Clean all output files: `make cleanAll`  
* Run individial test (take tmv for example): `cd tmv && make`  
  * If Test PASSED is printed, then test passes. Otherwise, if test fails, Test FAILED will be printed.<br>
* Clean individial test outputs (take tmv for example): `cd tmv && make clean`
* Run test case with the original naive kernel (take tmv for example): `cd tmv && make original`
* Profile the execution time for naive, coalesced, and prefetched kernels (take tmv for example): `cd tmv && make profile`
* To generate intermeidate ll files after each pass (take tmv for example): `cd tmv && make debug`
  * The ll for original naive kernel can be found in tmv.nvvm.ll
  * The ll for coalesced kernel can be found in tmv.coal.ll
  * The ll for prefetched kernel can be found in tmv.opt.ll
