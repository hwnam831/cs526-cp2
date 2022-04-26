LEVEL = ../..

## change or use make TARGET=inputfilename
#TARGET=square

## replace LLVMROOT as appropriate
LLVMROOT ?= $(HOME)/llvm
CUDAROOT = /usr/local/cuda

LLVMGCC = $(LLVMROOT)/bin/clang++
#LLI = $(LLVMROOT)/bin/lli
LLC = $(LLVMROOT)/bin/llc
LLVMAS  = $(LLVMROOT)/bin/llvm-as
LLVMDIS = $(LLVMROOT)/bin/llvm-dis
LLVMOPT = $(LLVMROOT)/bin/opt
OPTLIB ?= $(LEVEL)/build/gpumemopt/libGPUMEMOPT.so
SUBDIRS := vv transpose conv conv_16_16 demosaic imregionmax matmul mv reduction strsm tmv

define RUNFOREACH
    for DIR in $(SUBDIRS); do \
        echo $$DIR; \
		$(MAKE) -C $$DIR; \
    done
endef
define CLEANFOREACH
    for DIR in $(SUBDIRS); do \
        echo $$DIR; \
		$(MAKE) -C $$DIR clean; \
    done
endef

.PRECIOUS: %.nvvm.ll

## Other choices: test or comparecfe (these will be provided later)
default: test
NVCCFLAGS = --cuda-gpu-arch=sm_61 --cuda-path=$(CUDAROOT) -I$(CUDAROOT)/nvvm/include 
NVCCLIBS = -L$(CUDAROOT)/lib64 -lcudart_static -ldl -lrt -lpthread -lcuda -L$(CUDAROOT)/nvvm/lib64 -lnvvm -lcublas
CLCFLAGS = -emit-llvm -c -target -nvptx64-nvidial-nvcl -Dcl_clang_storage_class_specifiers\
 -I$(LLVMROOT)/include -include $(LLVMROOT)/include/clc/clcfunc.h -include $(LLVMROOT)/include/clc/clc.h -fpack-struct=64 \
 -Xclang -mlink-bitcode-file -Xclang $(LLVMROOT)/lib/clc/nvptx64--nvidiacl.bc -Dcl_khr_fp64  \
 -Xclang -fdeclare-opencl-builtins -Xclang -disable-O0-optnone -cl-std=CL2.0 -O1
DEBUGOPTS =  -verify -loop-simplify -licm -mem2reg -inline -adce -sccp 

test: $(TARGET)Loader $(TARGET).opt.ptx.s
	./$(TARGET)Loader $(TARGET).opt.ptx.s $(TARGET)

original: $(TARGET)Loader $(TARGET).nvvm.ptx.s
	./$(TARGET)Loader $(TARGET).nvvm.ptx.s $(TARGET)


profile: $(TARGET)Loader $(TARGET).nvvm.ptx.s $(TARGET).coal.ptx.s $(TARGET).opt.ptx.s
	nvprof ./$(TARGET)Loader $(TARGET).nvvm.ptx.s $(TARGET)
	nvprof ./$(TARGET)Loader $(TARGET).coal.ptx.s $(TARGET)
	nvprof ./$(TARGET)Loader $(TARGET).opt.ptx.s $(TARGET)

optimize: $(TARGET).opt.ll
debug: $(TARGET).debug.ll $(TARGET).nvvm.ll

all: 
	$(call RUNFOREACH)

cleanAll:
	$(call CLEANFOREACH)

gpumemopt: $(LEVEL)/gpumemopt/gpumemopt.cpp $(LEVEL)/gpumemopt/utils.h $(LEVEL)/gpumemopt/prefetch.cpp
	cmake --build ../../build	

%.ll: %.bc
	$(LLVMDIS) $<

$(TARGET)Loader: $(TARGET)Loader.cu
	$(LLVMGCC) $< -o $@ $(NVCCFLAGS) $(NVCCLIBS)

%.nvvm.bc: %_kernel.cl
	$(LLVMGCC) $< -o $@ $(CLCFLAGS)

%.debug.bc: %.nvvm.bc gpumemopt
	$(LLVMOPT) -load $(OPTLIB) --gpumemopt -o=$@ < $<

%.coal.bc: %.nvvm.bc gpumemopt
	$(LLVMOPT) -load $(OPTLIB) --gpumemopt -o=$@ < $<

%.opt.bc: %.coal.bc gpumemopt
	$(LLVMOPT) -load $(OPTLIB) --gpumempref -o=$@ < $<

%.ptx.s: %.bc
	$(LLVMOPT) -O2 $< | $(LLC) -mcpu=sm_61 -march=nvptx64 -o $@

clean:
	$(RM) -f $(TARGET)Loader *.bc *.ll *.ptx.s
