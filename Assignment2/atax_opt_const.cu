/* atax_const.cu - versione CUDA ATAX con constant memory e kernel ottimizzati */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

extern "C" {
    #include <polybench.h>
}
#include "atax.h"

#define BLOCK_SIZE 128

/* Constant memory per il vettore x */
__constant__ DATA_TYPE x_d_const[8192];

/* Simple CUDA error-check macro */
#define CUDA_CHECK(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      fprintf(stderr, "CUDA error at %s:%d -> %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(EXIT_FAILURE); \
    } \
  } while (0)

/* Kernel: tmp[i] = sum_j A[i][j] * x[j] (x in constant memory) */
__global__ void kernel_tmp_optimized(const DATA_TYPE* A, DATA_TYPE* tmp, int nx, int ny) {
    int row = blockIdx.x;              // 1 block per row
    if (row >= nx) return;

    int tx = threadIdx.x;
    __shared__ DATA_TYPE sdata[BLOCK_SIZE];

    DATA_TYPE partial_sum = 0.0;
    for (int j = tx; j < ny; j += blockDim.x)
        partial_sum += A[row * ny + j] * x_d_const[j];

    sdata[tx] = partial_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tx < stride) sdata[tx] += sdata[tx + stride];
        __syncthreads();
    }

    if (tx == 0) tmp[row] = sdata[0];
}

#define TILE_COLS 128
#define ROW_TILE 128

/* Kernel: y[j] = sum_i A[i][j] * tmp[i] */
__global__ void kernel_y_optimized(const DATA_TYPE* __restrict__ A,
                                   const DATA_TYPE* __restrict__ tmp,
                                   DATA_TYPE* y,
                                   int nx, int ny)
{
    const int col_start = blockIdx.x * TILE_COLS;
    const int tid = threadIdx.x;
    const int col = col_start + tid;
    if (col >= ny) return;

    DATA_TYPE sum = (DATA_TYPE)0.0;

    for (int row_base = 0; row_base < nx; row_base += ROW_TILE) {
        int row_end = row_base + ROW_TILE;
        if (row_end > nx) row_end = nx;
        for (int r = row_base; r < row_end; ++r)
            sum += A[r * ny + col] * tmp[r];
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

    /* Host arrays (PolyBench) */
    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NX, NY, nx, ny);
    POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, NY, ny);
    POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, NY, ny);
    POLYBENCH_1D_ARRAY_DECL(tmp, DATA_TYPE, NX, nx);

    init_array(nx, ny, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(x));

    /* Raw host pointers */
    DATA_TYPE* A_h = &POLYBENCH_ARRAY(A)[0][0];
    DATA_TYPE* x_h = &POLYBENCH_ARRAY(x)[0];
    DATA_TYPE* y_h = &POLYBENCH_ARRAY(y)[0];
    DATA_TYPE* tmp_h = &POLYBENCH_ARRAY(tmp)[0];

    /* Device pointers */
    DATA_TYPE *A_d = NULL, *y_d = NULL, *tmp_d = NULL;

    size_t sizeA = (size_t)nx * (size_t)ny * sizeof(DATA_TYPE);
    size_t sizey = (size_t)ny * sizeof(DATA_TYPE);
    size_t sizetmp = (size_t)nx * sizeof(DATA_TYPE);
    size_t sizex = (size_t)ny * sizeof(DATA_TYPE);

    /* Constant memory size check */
    if (ny > 8192) {
        fprintf(stderr, "Error: ny (%d) > constant buffer (8192)\n", ny);
        return EXIT_FAILURE;
    }

    /* Device allocations */
    CUDA_CHECK(cudaMalloc((void**)&A_d, sizeA));
    CUDA_CHECK(cudaMalloc((void**)&y_d, sizey));
    CUDA_CHECK(cudaMalloc((void**)&tmp_d, sizetmp));

    /* Copy inputs: A -> global, x -> constant */
    CUDA_CHECK(cudaMemcpy(A_d, A_h, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(x_d_const, x_h, sizex));

    /* Timing + kernel launches */
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));

    dim3 grid_tmp(nx);
    dim3 block_tmp(BLOCK_SIZE);
    kernel_tmp_optimized<<<grid_tmp, block_tmp>>>(A_d, tmp_d, nx, ny);
    CUDA_CHECK(cudaGetLastError());

    dim3 grid_y((ny + TILE_COLS - 1) / TILE_COLS);
    dim3 block_y(TILE_COLS);
    kernel_y_optimized<<<grid_y, block_y>>>(A_d, tmp_d, y_d, nx, ny);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("**************************************************\n");
    printf("GPU kernels elapsed time (constant memory - TILING): %f ms\n", milliseconds);
    printf("**************************************************\n");

    /* Copy result back */
    CUDA_CHECK(cudaMemcpy(y_h, y_d, sizey, cudaMemcpyDeviceToHost));

    /* Cleanup */
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(y_d));
    CUDA_CHECK(cudaFree(tmp_d));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(x);
    POLYBENCH_FREE_ARRAY(y);
    POLYBENCH_FREE_ARRAY(tmp);

    return 0;
}
