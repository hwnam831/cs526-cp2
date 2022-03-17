# cs526-cp2
./build/bin/clang-13 -c -x cl -emit-llvm -S -cl-std=CL2.0 kernel.cl -o kernel.ll
./build/bin/llc -march=nvptx64 kernel.ll -o kernel.ptx
./build/bin/clang++ sample.cpp -o sample -g -I /usr/local/cuda/include -lcuda

## Requirements
CUDA installed at `/usr/local/cuda`  
`sudo apt install mesa-common-dev`  
llvm installed or linked to `./llvm`, or set Makefile's LLVMROOT accordingly  
libclc (https://github.com/llvm-mirror/libclc) built (only nvptx) and installed to LLVMROOT  
Example configure script (change --prefix and --with-llvm-config accordingly):  
`./configure.py --prefix=/home/hwnam/llvm --with-llvm-config=/home/hwnam/llvm/bin/llvm-config nvptx--nvidiacl nvptx64--nvidiacl`
