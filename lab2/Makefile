ifndef CUDA_HOME
CUDA_HOME:=/usr/local/cuda
endif

# Nome file .cu del benchmark (modificabile)
EXERCISE=atax.cu

BUILD_DIR ?= ./build

NVCC=$(CUDA_HOME)/bin/nvcc
CXX=g++

OPT:=-O3
# Mantengo alcune flags della esercitazione; modifica -arch=sm_53 se serve
NVOPT:=-lineinfo -arch=sm_53 --ptxas-options=-v --use_fast_math

CXXFLAGS:=$(OPT) -I. -I../../../../utilities $(EXT_CXXFLAGS)
LDFLAGS:=-lm $(EXT_LDFLAGS)

NVCFLAGS:=$(CXXFLAGS) $(NVOPT)
NVLDFLAGS:=$(LDFLAGS) -lcudart

SRCS:= ../../../../utilities/polybench.c
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o) $(EXERCISE:%=$(BUILD_DIR)/%.o)
EXE=atax_acc.exe

$(EXE):	$(OBJS)
	$(MKDIR_P) $(dir $@)
	$(NVCC) $(NVCFLAGS) $(OBJS) -o $@ $(NVLDFLAGS)

$(BUILD_DIR)/%.cu.o: %.cu
	$(MKDIR_P) $(dir $@)
	$(NVCC) $(NVCFLAGS) -c $< -o $@

$(BUILD_DIR)/%.c.o: %.c
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

all: $(EXE)

.PHONY: run profile clean

run: $(EXE)
	./$(EXE)

profile: $(EXE)
	$(CUDA_HOME)/bin/nvprof ./$(EXE)

clean:
	-rm -fr $(BUILD_DIR) *.exe *.out *~

MKDIR_P ?= mkdir -p

