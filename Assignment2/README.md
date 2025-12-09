## Overview

This project systematically evaluates different CUDA memory management strategies and GPU optimization techniques to achieve measurable speedup on NVIDIA GPUs. The baseline CUDA implementation is progressively enhanced using pinned memory, unified memory, constant memory caching, shared memory tiling, and CUDA streams for compute/transfer overlap.

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

### Optimization Variants in `atax.cu`

The main `atax.cu` file supports multiple GPU optimization strategies via preprocessor flags:

1. **`BASE`** – Baseline CUDA implementation with `cudaMalloc` and `cudaMemcpy`
2. **`PINNED`** – Pinned (page-locked) host memory for faster PCIe transfers
3. **`UVM`** – Unified Memory (managed memory) with automatic migration
4. **`CONST`** – Constant memory caching for read-only vector `x`
5. **`TILING`** – Shared memory tiling with reduction trees for improved occupancy
6. **`TRANSPOSE`** – Matrix transposition on GPU for optimized memory access patterns
7. **`STREAMS`** – CUDA streams for overlapping data transfers and kernel execution
8. **`OPT_STREAMS`** – Combined streams + tiling optimizations

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

- NVIDIA GPU with CUDA support (compute capability 5.0+)
- CUDA Toolkit (11.0 or later)
- nvcc compiler
- PolyBench/C framework (included via headers)

### Automated Benchmarking

Run the `run.sh` script to compile and execute all variants across all dataset sizes:

```bash
bash run.sh
```

This will:
1. Compile each kernel variant with each dataset size
2. Execute and capture timing results
3. Store all outputs in `atax_timings.csv`

## Key Optimization Techniques

### 1. Pinned Memory
We managed to use `cudaMallocHost` to allocate additional space in CPU RAM that will be faster to transfer over PICe bus to device memory. 

### 2. Unified Memory (UVM)
We managet to use `cudaMallocManaged` to allocate memory that is shared between CPU and GPU, allowing the CUDA runtime to handle data migration automatically.
### 3. Constant Memory
In atax case, constant memory is used to keep the vector x, which is read-only during kernel execution. This allows faster access time due to caching.
We also had to check for x dimensions to not exeed constant memory size limit (64KB).
### 4. Shared Memory Tiling

### 5. Matrix Transposition

### 6. CUDA Streams
Streams in cuda allow to overlap data transfers and kernel execution to improve performance. In this way we can keep CPU and GPU both buty at the same time creating a sort of parallel work pipeline between the two of them.
We avoided the creation of too many streams due to overhead in thei management.
This technique also require pinned memory to work due to memory synchronization issues.

### 7. Combined Optimizations
In the `OPT_STREAMS` variant, we combined streams with shared memory tiling to maximize performance gains.


## Results

Results can be found in the 'plots' directory, showing performance comparisons across different optimization strategies and dataset sizes, including speedup curves relative to the baseline CUDA implementation.

## License

This project is part of a High Performance Computing course assignment.

---

**Authors**: Alessandro Chiarabini, Massimiliano Nardone, Manuel Testoni - HPC Lab Course, 4th Year  
**Institution**: University of Modena and Reggio Emilia   
**Date**: December 2025
