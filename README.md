# cs526-cp2

## Requirements
CUDA installed at `/usr/local/cuda`  
`sudo apt install mesa-common-dev`  
llvm installed or linked to `./llvm`, or set Makefile's LLVMROOT accordingly  
libclc (https://github.com/llvm-mirror/libclc) built (only nvptx) and installed to LLVMROOT  
Example configure script (change --prefix and --with-llvm-config accordingly):  
`./configure.py --prefix=/home/hwnam/llvm --with-llvm-config=/home/hwnam/llvm/bin/llvm-config nvptx--nvidiacl nvptx64--nvidiacl`
