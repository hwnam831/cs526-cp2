LEVEL = ../..

## change or use make TARGET=inputfilename
TARGET=square

## replace LLVMROOT as appropriate
LLVMROOT ?= ~/llvm
CUDAROOT = /usr/local/cuda

LLVMGCC = $(LLVMROOT)/bin/clang++
#LLI = $(LLVMROOT)/bin/lli
LLC = $(LLVMROOT)/bin/llc
LLVMAS  = $(LLVMROOT)/bin/llvm-as
LLVMDIS = $(LLVMROOT)/bin/llvm-dis
LLVMOPT = $(LLVMROOT)/bin/opt



## Other choices: test or comparecfe (these will be provided later)
default: test
NVCCFLAGS = --cuda-gpu-arch=sm_61 --cuda-path=$(CUDAROOT) -I$(CUDAROOT)/nvvm/include 
NVCCLIBS = -L$(CUDAROOT)/lib64 -lcudart_static -ldl -lrt -lpthread -lcuda -L$(CUDAROOT)/nvvm/lib64 -lnvvm
CLCFLAGS = -emit-llvm -c -target -nvptx64-nvidial-nvcl -Dcl_clang_storage_class_specifiers\
 -I$(LLVMROOT)/include -include $(LLVMROOT)/include/clc/clc.h -fpack-struct=64 \
 -Xclang -mlink-bitcode-file -Xclang $(LLVMROOT)/lib/clc/nvptx64--nvidiacl.bc -Dcl_khr_fp64  \
 -Xclang -fdeclare-opencl-builtins -cl-std=CL2.0

test: loader $(TARGET).ptx.s
	./loader $(TARGET).ptx.s $(TARGET)

.PRECIOUS: %.ll


%.nvvm.ll: %.nvvm.bc
	$(LLVMDIS) $<

loader: loader.cu
	$(LLVMGCC) $< -o $@ $(NVCCFLAGS) $(NVCCLIBS)

%.nvvm.bc: %.cl
	$(LLVMGCC) $< -o $@ $(CLCFLAGS)

%.ptx.s: %.nvvm.ll
	$(LLC) -mcpu=sm_61 -march=nvptx64 $< -o $@

clean:
	$(RM) -f loader *.bc *.ll *.ptx.s
