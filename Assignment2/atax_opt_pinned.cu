/* atax_pinned.cu - ATAX CUDA con host pinned + device buffers (corretto) */

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

/* ------------------------------------------------------------------
   Kernels (usano puntatori device). Mantieni i tuoi kernels.
   ------------------------------------------------------------------*/

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

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
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

/* Initialize arrays (host) */
static void init_array_host(int nx, int ny, DATA_TYPE *A_h, DATA_TYPE *x_h)
{
    for (int i = 0; i < ny; i++) x_h[i] = i * M_PI;
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            A_h[i * ny + j] = ((DATA_TYPE)i * (j + 1)) / nx;
}

/* Optional print for verification */
static void print_array_host(int ny, DATA_TYPE *y_h)
{
    for (int i = 0; i < ny; i++) {
        fprintf(stderr, DATA_PRINTF_MODIFIER, y_h[i]);
        if ((i+1) % 20 == 0) fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");
}

/* Main: pinned host buffers + device buffers */
int main(int argc, char** argv) {
    int nx = NX;
    int ny = NY;

    size_t sizeA = (size_t)nx * (size_t)ny * sizeof(DATA_TYPE);
    size_t sizex = (size_t)ny * sizeof(DATA_TYPE);
    size_t sizey = (size_t)ny * sizeof(DATA_TYPE);
    size_t sizetmp = (size_t)nx * sizeof(DATA_TYPE);

    /* --- Host pinned buffers (for faster H2D/D2H) --- */
    DATA_TYPE *A_h_pinned = NULL, *x_h_pinned = NULL, *y_h_pinned = NULL, *tmp_h_pinned = NULL;
    CUDA_CHECK(cudaMallocHost((void**)&A_h_pinned, sizeA));    // pinned host
    CUDA_CHECK(cudaMallocHost((void**)&x_h_pinned, sizex));    // pinned host
    CUDA_CHECK(cudaMallocHost((void**)&y_h_pinned, sizey));    // pinned host (to receive result)
    CUDA_CHECK(cudaMallocHost((void**)&tmp_h_pinned, sizetmp)); // optional pinned host for tmp

    /* Initialize host pinned arrays */
    init_array_host(nx, ny, A_h_pinned, x_h_pinned);
    // no need to initialize y_h_pinned/tmp_h_pinned - kernels will write result

    /* --- Device buffers (GPU) --- */
    DATA_TYPE *A_d = NULL, *x_d = NULL, *y_d = NULL, *tmp_d = NULL;
    CUDA_CHECK(cudaMalloc((void**)&A_d, sizeA));
    CUDA_CHECK(cudaMalloc((void**)&x_d, sizex));
    CUDA_CHECK(cudaMalloc((void**)&y_d, sizey));
    CUDA_CHECK(cudaMalloc((void**)&tmp_d, sizetmp));

    /* --- Copy host (pinned) -> device --- */
    CUDA_CHECK(cudaMemcpy(A_d, A_h_pinned, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(x_d, x_h_pinned, sizex, cudaMemcpyHostToDevice));
    // y_d and tmp_d are written by kernels

    /* --- Launch kernels and time --- */
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));

    dim3 grid_tmp(nx);
    dim3 block_tmp(THREADS_PER_BLOCK);
    kernel_tmp_optimized<<<grid_tmp, block_tmp>>>(A_d, x_d, tmp_d, nx, ny);
    CUDA_CHECK(cudaGetLastError());

    dim3 grid_y((ny + TILE_COLS - 1) / TILE_COLS);
    dim3 block_y(TILE_COLS);
    kernel_y_optimized<<<grid_y, block_y>>>(A_d, tmp_d, y_d, nx, ny);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("**************************************************\n");
    printf("GPU kernels elapsed time (pinned memory - TILING): %f ms\n", ms);
    printf("**************************************************\n");

    /* --- Copy result device -> pinned host --- */
    CUDA_CHECK(cudaMemcpy(y_h_pinned, y_d, sizey, cudaMemcpyDeviceToHost));

    /* Optional: verify / copy back to polybench arrays if needed */
    // If you still use POLYBENCH arrays, copy pinned host to them:
    // POLYBENCH_2D_ARRAY_DECL/1D_ARRAY_DECL not used here; if used, memcpy to them.

    /* Example: print first 10 results */
    // print_array_host(ny, y_h_pinned);

    /* --- Cleanup --- */
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(x_d));
    CUDA_CHECK(cudaFree(y_d));
    CUDA_CHECK(cudaFree(tmp_d));

    CUDA_CHECK(cudaFreeHost(A_h_pinned));
    CUDA_CHECK(cudaFreeHost(x_h_pinned));
    CUDA_CHECK(cudaFreeHost(y_h_pinned));
    CUDA_CHECK(cudaFreeHost(tmp_h_pinned));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
/* ------------------------------------------------------------------
   End of kernels and main
   ------------------------------------------------------------------*/