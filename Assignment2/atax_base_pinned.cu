/* atax.cu - versione CUDA del kernel ATAX
   Assunzione: atax.h e polybench.h sono quelli di PolyBench (come nel tuo progetto).
   Compilare con nvcc (Makefile fornito sotto).
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

extern "C" {
    #include <polybench.h>
}
#include "atax.h"

#define BLOCK_SIZE 256

/* Simple CUDA error-check macro */
#define CUDA_CHECK(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      fprintf(stderr, "CUDA error at %s:%d -> %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(EXIT_FAILURE); \
    } \
  } while (0)

/* Kernel: compute tmp[i] = sum_j A[i][j] * x[j]  (one thread per row i) */
__global__ void kernel_tmp(const DATA_TYPE* A, const DATA_TYPE* x, DATA_TYPE* tmp, int nx, int ny) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nx) {
    DATA_TYPE sum = 0.0;
    int row_off = i * ny;
    for (int j = 0; j < ny; ++j) {
      sum += A[row_off + j] * x[j];
    }
    tmp[i] = sum;
  }
}

/* Kernel: compute y[j] = sum_i A[i][j] * tmp[i]  (one thread per column j) */
__global__ void kernel_y(const DATA_TYPE* A, const DATA_TYPE* tmp, DATA_TYPE* y, int nx, int ny) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < ny) {
    DATA_TYPE sum = 0.0;
    for (int i = 0; i < nx; ++i) {
      sum += A[i * ny + j] * tmp[i];
    }
    y[j] = sum;
  }
}

static void init_array(int nx, int ny,
                       DATA_TYPE POLYBENCH_2D(A, NX, NY, nx, ny),
                       DATA_TYPE POLYBENCH_1D(x, NY, ny))
{
  int i, j;
  for (i = 0; i < ny; i++)
    x[i] = i * M_PI;
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++)
      A[i][j] = ((DATA_TYPE)i * (j + 1)) / nx;
}

static void print_array(int nx,
                        DATA_TYPE POLYBENCH_1D(y, NX, nx))
{
  int i;
  for (i = 0; i < nx; i++) {
    fprintf(stderr, DATA_PRINTF_MODIFIER, y[i]);
    if (i % 20 == 0) fprintf(stderr, "\n");
  }
  fprintf(stderr, "\n");
}

int main(int argc, char** argv) {
  int nx = NX;
  int ny = NY;

  /* Declare + allocate host arrays using polybench macros */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NX, NY, nx, ny);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, NY, ny);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, NY, ny);
  POLYBENCH_1D_ARRAY_DECL(tmp, DATA_TYPE, NX, nx);

  /* Initialize host arrays */
  init_array(nx, ny, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(x));

  /* Pointers to host raw memory (row-major) */
  DATA_TYPE* A_h = &POLYBENCH_ARRAY(A)[0][0];
  DATA_TYPE* x_h = &POLYBENCH_ARRAY(x)[0];
  DATA_TYPE* y_h = &POLYBENCH_ARRAY(y)[0];
  DATA_TYPE* tmp_h = &POLYBENCH_ARRAY(tmp)[0];

  /* Device pointers */
  DATA_TYPE *A_d = NULL, *x_d = NULL, *y_d = NULL, *tmp_d = NULL;

  size_t sizeA = (size_t)nx * (size_t)ny * sizeof(DATA_TYPE);
  size_t sizex = (size_t)ny * sizeof(DATA_TYPE);
  size_t sizey = (size_t)ny * sizeof(DATA_TYPE);
  size_t sizetmp = (size_t)nx * sizeof(DATA_TYPE);

  /* Allocate device memory */
  CUDA_CHECK(cudaMallocHost((void**)&A_d, sizeA));
  CUDA_CHECK(cudaMallocHost((void**)&x_d, sizex));
  CUDA_CHECK(cudaMallocHost((void**)&y_d, sizey));
  CUDA_CHECK(cudaMallocHost((void**)&tmp_d, sizetmp));

  /* Copy inputs to device */
  CUDA_CHECK(cudaMemcpy(A_d, A_h, sizeA, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(x_d, x_h, sizex, cudaMemcpyHostToDevice));
  /* No need to init y_d or tmp_d (kernels write them) */

  /* Launch kernels and measure time with cudaEvent */
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, 0));

  /* Compute tmp */
  int block = BLOCK_SIZE;
  int grid_tmp = (nx + block - 1) / block;
  kernel_tmp<<<grid_tmp, block>>>(A_d, x_d, tmp_d, nx, ny);
  CUDA_CHECK(cudaGetLastError());

  /* Compute y */
  int grid_y = (ny + block - 1) / block;
  kernel_y<<<grid_y, block>>>(A_d, tmp_d, y_d, nx, ny);
  CUDA_CHECK(cudaGetLastError());

  /* Synchronize and stop timer */
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(stop, 0));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float milliseconds = 0;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  printf("**************************************************\n");
  printf("GPU kernels elapsed time (BASE - PINNED): %f ms\n", milliseconds);
  printf("**************************************************\n");

  /* Copy result back */
  CUDA_CHECK(cudaMemcpy(y_h, y_d, sizey, cudaMemcpyDeviceToHost));

  /* Print results to prevent DCE (use existing print_array) */
  //print_array(nx, POLYBENCH_ARRAY(y));

  /* Free device memory */
  CUDA_CHECK(cudaFreeHost(A_d));
  CUDA_CHECK(cudaFreeHost(x_d));
  CUDA_CHECK(cudaFreeHost(y_d));
  CUDA_CHECK(cudaFreeHost(tmp_d));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  /* Free host arrays (polybench macros) */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(y);
  POLYBENCH_FREE_ARRAY(tmp);

  return 0;
}