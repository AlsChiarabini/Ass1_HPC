/* atax_uvm.cu - versione CUDA ATAX con Unified Memory (UVM) */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

extern "C" {
    #include <polybench.h>
}
#include "atax.h"

#define BLOCK_SIZE 128
#define THREADS_PER_BLOCK 128
#define TILE_COLS 128
#define ROW_TILE 128

/* Simple CUDA error-check macro */
#define CUDA_CHECK(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      fprintf(stderr, "CUDA error at %s:%d -> %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(EXIT_FAILURE); \
    } \
  } while (0)

/* Kernel: compute tmp[i] = sum_j A[i][j] * x[j] */
__global__ void kernel_tmp_optimized(const DATA_TYPE* A, const DATA_TYPE* x, DATA_TYPE* tmp, int nx, int ny) {
    int row = blockIdx.x;
    if (row >= nx) return;

    int tx = threadIdx.x;
    __shared__ DATA_TYPE sdata[THREADS_PER_BLOCK];

    DATA_TYPE partial_sum = 0.0;
    for (int j = tx; j < ny; j += blockDim.x)
        partial_sum += A[row * ny + j] * x[j];

    sdata[tx] = partial_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tx < stride) sdata[tx] += sdata[tx + stride];
        __syncthreads();
    }

    if (tx == 0) tmp[row] = sdata[0];
}

/* Kernel: compute y[j] = sum_i A[i][j] * tmp[i] */
__global__ void kernel_y_optimized(const DATA_TYPE* __restrict__ A,
                                   const DATA_TYPE* __restrict__ tmp,
                                   DATA_TYPE* y,
                                   int nx, int ny)
{
    const int col_start = blockIdx.x * TILE_COLS;
    int tid = threadIdx.x;
    int col = col_start + tid;
    if (col >= ny) return;

    DATA_TYPE sum = 0.0;
    for (int row_base = 0; row_base < nx; row_base += ROW_TILE) {
        int row_end = row_base + ROW_TILE;
        if (row_end > nx) row_end = nx;

        for (int r = row_base; r < row_end; ++r)
            sum += A[(size_t)r * (size_t)ny + (size_t)col] * tmp[r];
    }
    y[col] = sum;
}

/* Initialize arrays */
static void init_array(int nx, int ny,
                       DATA_TYPE POLYBENCH_2D(A, NX, NY, nx, ny),
                       DATA_TYPE POLYBENCH_1D(x, NY, ny))
{
    for (int i = 0; i < ny; i++) x[i] = i * M_PI;
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            A[i][j] = ((DATA_TYPE)i * (j + 1)) / nx;
}

int main(int argc, char** argv) {
    int nx = NX;
    int ny = NY;

    /* Device pointers (UVM) */
    DATA_TYPE *A, *x, *y, *tmp;
    CUDA_CHECK(cudaMallocManaged(&A, nx * ny * sizeof(DATA_TYPE)));
    CUDA_CHECK(cudaMallocManaged(&x, ny * sizeof(DATA_TYPE)));
    CUDA_CHECK(cudaMallocManaged(&y, ny * sizeof(DATA_TYPE)));
    CUDA_CHECK(cudaMallocManaged(&tmp, nx * sizeof(DATA_TYPE)));

    /* Initialize arrays directly in UVM (CPU accessible) */
    for (int i = 0; i < ny; i++) x[i] = i * M_PI;
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            A[i * ny + j] = ((DATA_TYPE)i * (j + 1)) / nx;

    /* Launch kernels with timing */
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));

    dim3 grid_tmp(nx), block_tmp(THREADS_PER_BLOCK);
    kernel_tmp_optimized<<<grid_tmp, block_tmp>>>(A, x, tmp, nx, ny);
    CUDA_CHECK(cudaGetLastError());

    dim3 grid_y((ny + TILE_COLS - 1) / TILE_COLS), block_y(TILE_COLS);
    kernel_y_optimized<<<grid_y, block_y>>>(A, tmp, y, nx, ny);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("**************************************************\n");
    printf("GPU kernels elapsed time (uvm memory - TILING): %f ms\n", milliseconds);
    printf("**************************************************\n");

    /* Free UVM memory */
    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(x));
    CUDA_CHECK(cudaFree(y));
    CUDA_CHECK(cudaFree(tmp));

    return 0;
}
