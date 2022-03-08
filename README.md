# cs526-cp2
./build/bin/clang-13 -c -x cl -emit-llvm -S -cl-std=CL2.0 kernel.cl -o kernel.ll
./build/bin/llc -march=nvptx64 kernel.ll -o kernel.ptx
./build/bin/clang++ sample.cpp -o sample -g -I /usr/local/cuda/include -lcuda
