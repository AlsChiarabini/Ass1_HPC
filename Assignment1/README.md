## Overview

This project systematically evaluates different parallelization and optimization strategies to achieve measurable speedup on multi-core CPUs and accelerators. The baseline sequential implementation is progressively enhanced using OpenMP directives, memory access optimizations, and GPU offloading.

## Problem Description

The ATAX kernel performs the following computation:

```
tmp = A × x        (matrix-vector multiplication)
y = A^T × tmp      (transpose-matrix-vector multiplication)
```

Where:
- **A** is an NX × NY matrix
- **x** is a vector of size NY
- **tmp** is a temporary vector of size NX
- **y** is the output vector of size NY

### Optimization Variants in `atax.c`

The main `atax.c` file supports multiple optimization strategies via preprocessor flags:

1. **`SEQUENTIAL`** – Baseline implementation without parallelization
2. **`PARALLEL`** – Basic OpenMP parallelization with `#pragma omp parallel for`
3. **`PARALLEL_NORACE`** – Thread-private accumulators to eliminate race conditions
4. **`REDUCTION`** – OpenMP array reductions for cleaner parallel aggregation
5. **`COLLAPSE`** – Loop collapsing to increase parallelism granularity
6. **`OPTIMIZED`** – SIMD vectorization + static scheduling for better cache utilization
7. **`OPTIMIZED_TILING`** – Cache-blocking (tiling) combined with SIMD and reductions
8. **`TARGET`** – GPU offloading with OpenMP target directives (same as `atax_target.c`)

## Dataset Sizes

Five problem sizes are supported via compile-time defines:

| Dataset            | Matrix Size (NX × NY) |
|--------------------|-----------------------|
| `MINI_DATASET`     | 32 × 32               |
| `SMALL_DATASET`    | 500 × 500             |
| `STANDARD_DATASET` | 4000 × 4000 (default) |
| `LARGE_DATASET`    | 8000 × 8000           |
| `EXTRALARGE_DATASET`| 20000 × 20000        |

## Building and Running

### Prerequisites

- A C compiler with OpenMP support (e.g., GCC, Clang)
- PolyBench/C framework (included via headers)


### Automated Benchmarking

Run the `run.sh` script to compile and execute all variants across all dataset sizes:

```bash
bash run.sh
```

This will:
1. Compile each kernel variant with each dataset size
2. Execute and capture timing results
3. Store all outputs in `atax_timing.csv`


## Key Optimization Techniques

### 1. Parallelization
Adding some "pragma omp parallel for" directives to let compiler know that there is the need to parallelize the loops.

### 2. Race Condition Elimination
Eliminating race conditions means using a thread-private accumulator for each thread. This will be added to the shared variable only once all the thread computations are done.

### 3. Memory Access Optimization
Changing the memory access pattern to ensure contiguous memory accesses, improving cache performance during matrix-vector multiplications.

### 4. SIMD Vectorization
SIMD (Single Instruction, Multiple Data) vectorization allows multiple data points to be processed simultaneously using vector registers. We utilized compiler auto-vectorization features and ensured data alignment for optimal performance.

### 5. Cache Blocking (Tiling)
Diving the matrix into smaller tiles to improve cache locality during matrix-vector multiplications. Compilers doesn not have an automatic tiling feature, so we manually implemented this optimization.

### 6. GPU Offloading
Moving the main computation of the program, in our case the two matrix-vector multiplications, to the GPU using OpenMP target directives. This involves: 
- copying data to/from the device
- managing device memory


## Results
Results can be found in the 'plot' directory. Inside this directory is it possible to find two main folders: 
- gcc
- clang
Because we had to compile each version twice in order to effectively use GPU.
There you can find istograms and speedup plots for each optimization technique showing which optimizations yielded to the major performance improvements.



## License

This project is part of a High Performance Computing course assignment.

---

**Authors**: Alessandro Chiarabini, Massimiliano Nardone, Manuel Testoni - HPC Lab Course, 4th Year  
**Institution**: University of Modena and Reggio Emilia   
**Date**: November 2025
