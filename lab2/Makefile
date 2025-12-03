# --- Variabili di Sistema e Percorsi ---

# Nuovo percorso per le utility di PolyBench (4 livelli di directory, non 3)
UTIL_DIR_PATH = ../../../../utilities

# Include i file di configurazione con il percorso corretto
-include $(UTIL_DIR_PATH)/options.mk
-include $(UTIL_DIR_PATH)/c2.mk
# Nota: L'uso di $(UTIL_DIR_PATH) previene problemi di path con la variabile UTIL_DIR definita in c2.mk

# --- Percorsi CUDA e Build (dal laboratorio) ---
ifndef CUDA_HOME
CUDA_HOME:=/usr/local/cuda
endif

BUILD_DIR ?= ./build

# --- Naming del Progetto ATAX ---
BENCHMARK = atax
EXERCISE = atax.cu
EXE = $(BENCHMARK)_cuda

# --- File Sorgente ---
# atax.cu è locale. polybench.c è remoto.
SRCS_UTIL = $(UTIL_DIR_PATH)/polybench.c
SRCS = $(SRCS_UTIL)
HEADERS = $(BENCHMARK).h

# --- Compilatori e Percorsi ---
NVCC=$(CUDA_HOME)/bin/nvcc
CC=gcc

# --- Flag e Opzioni ---

# FLAG ESTERNI per PolyBench e Profiling
# Useremo sempre questa variabile al posto di EXT_CXXFLAGS/EXT_CFLAGS
EXT_FLAGS = -DLARGE_DATASET -DPOLYBENCH_TIME -pg $(EXT_CFLAGS)

# OPT: flag base
OPT:=-O2 -g

# NVOPT: Flag specifici per nvcc
# VEDI ATTENZIONE SOTTO: Ho impostato sm_80 come ipotesi moderna.
NVOPT:=-Xcompiler "-fopenmp" -lineinfo -arch=sm_50 --ptxas-options=-v --use_fast_math

# CFLAGS: Flag per il codice C (per polybench.c)
CFLAGS:=$(OPT) $(EXT_FLAGS) -I. -I$(UTIL_DIR_PATH)

# NVCFLAGS: Flag completi per NVCC
NVCFLAGS:=$(OPT) $(EXT_FLAGS) $(NVOPT) -I. -I$(UTIL_DIR_PATH)

# NVLDFLAGS: Flag completi di linking per NVCC
NVLDFLAGS:=-lm -lcudart -lgomp -ccbin gcc $(EXT_LDFLAGS)

# --- Oggetti (.o) e Dipendenze ---
# Estrai solo il nome del file (polybench.c) e crea l'oggetto nella build dir
POLYBENCH_FILENAME = $(notdir $(SRCS_UTIL))

# Oggetti C (solo polybench.c) -> Esempio: build/polybench.o
OBJS_UTIL = $(BUILD_DIR)/$(POLYBENCH_FILENAME:.c=.o)
# Oggetto CUDA (atax.cu) -> Esempio: build/atax.cu.o
OBJS_CUDA = $(EXERCISE:%=$(BUILD_DIR)/%.o)
# Lista completa di tutti gli oggetti
OBJS = $(OBJS_UTIL) $(OBJS_CUDA)

DEPS      := Makefile.dep
DEP_FLAG  := -MM
MKDIR_P ?= mkdir -p

# --- Regole di Compilazione ---

.PHONY: all exe clean veryclean run profile

all: $(EXE)

# Regola di LINKING: Usa NVCC
$(EXE): $(OBJS)
	$(MKDIR_P) $(dir $(BUILD_DIR))
	$(NVCC) $(NVCFLAGS) $(OBJS) -o $@ $(NVLDFLAGS)

# *** REGOLA ESPLICITA PER IL FILE C REMOTO (POLYBENCH.C) ***
# Questa regola gestisce il file C che non è nella cartella locale
$(OBJS_UTIL): $(SRCS_UTIL)
	$(MKDIR_P) $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

# Regola per i file CUDA (.cu -> .o) - atax.cu (in cartella locale)
$(BUILD_DIR)/%.cu.o: %.cu
	$(MKDIR_P) $(dir $@)
	$(NVCC) $(NVCFLAGS) -c $< -o $@

# --- Regole di Pulizia e Esecuzione ---
clean:
	-rm -fr $(BUILD_DIR) $(EXE) *.out *~

veryclean : clean
	-rm -vf $(DEPS)

run: $(EXE)
	./$(EXE)

profile: $(EXE)
	sudo $(CUDA_HOME)/bin/nvprof ./$(EXE)

# --- Regole per le Dipendenze ---
$(DEPS): $(SRCS) $(EXERCISE) $(HEADERS)
	$(NVCC) $(CFLAGS) $(DEP_FLAG) $(SRCS) $(EXERCISE) > $(DEPS)

-include $(DEPS)
