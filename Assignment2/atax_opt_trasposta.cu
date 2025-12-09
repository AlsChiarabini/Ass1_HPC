/* atax_gpu_transpose.cu - ATAX CUDA con trasposta interamente GPU */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

extern "C" {
    #include <polybench.h>
}
#include "atax.h"

#define BLOCK_SIZE 256
#define TILE_SIZE 16   // tiling per kernel di trasposizione

#define CUDA_CHECK(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      fprintf(stderr, "CUDA error at %s:%d -> %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(EXIT_FAILURE); \
    } \
  } while (0)

/* Kernel tmp ottimizzato (1 blocco = 1 riga, riduzione intra-block) */
__global__ void kernel_tmp_optimized(const DATA_TYPE* A, const DATA_TYPE* x, DATA_TYPE* tmp, int nx, int ny) {
    int row = blockIdx.x;
    if (row >= nx) return;

    int tx = threadIdx.x;
    __shared__ DATA_TYPE sdata[BLOCK_SIZE];

    DATA_TYPE partial_sum = 0.0;
    for (int j = tx; j < ny; j += blockDim.x)
        partial_sum += A[row * ny + j] * x[j];

    sdata[tx] = partial_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tx < stride)
            sdata[tx] += sdata[tx + stride];
        __syncthreads();
    }

    if (tx == 0)
        tmp[row] = sdata[0];
}

/* Kernel y ottimizzato (1 blocco = 1 colonna di A_T) */
__global__ void kernel_y_optimized(const DATA_TYPE* A_T, const DATA_TYPE* tmp, DATA_TYPE* y, int nx, int ny) {
    int col = blockIdx.x;
    if (col >= ny) return;

    int tid = threadIdx.x;
    __shared__ DATA_TYPE sdata[BLOCK_SIZE];

    DATA_TYPE sum = 0.0;
    for (int t = 0; t < nx; t += BLOCK_SIZE) {
        int tileLimit = min(BLOCK_SIZE, nx - t);
        sdata[tid] = 0.0;

        // Accumulo parziale per questo tile
        for (int i = tid; i < tileLimit; i += blockDim.x)
            sdata[tid] += A_T[col * nx + (t + i)] * tmp[t + i];

        __syncthreads();

        // Riduzione corretta: controllo SOLO tid < stride
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (tid < stride)
                sdata[tid] += sdata[tid + stride];
            __syncthreads();
        }

        if (tid == 0)
            sum += sdata[0];

        __syncthreads();
    }

    if (tid == 0)
        y[col] = sum;
}


/* Kernel per trasporre A → A_T usando tiling in shared memory */
__global__ void transpose_kernel(const DATA_TYPE* A, DATA_TYPE* A_T, int nx, int ny) {
    __shared__ DATA_TYPE tile[TILE_SIZE][TILE_SIZE+1];

    int xIndex = blockIdx.x * TILE_SIZE + threadIdx.x;
    int yIndex = blockIdx.y * TILE_SIZE + threadIdx.y;

    if (xIndex < ny && yIndex < nx)
        tile[threadIdx.y][threadIdx.x] = A[yIndex * ny + xIndex];

    __syncthreads();

    int transposed_x = blockIdx.y * TILE_SIZE + threadIdx.x;
    int transposed_y = blockIdx.x * TILE_SIZE + threadIdx.y;

    if (transposed_x < nx && transposed_y < ny)
        A_T[transposed_y * nx + transposed_x] = tile[threadIdx.x][threadIdx.y];
}

/* Initialize arrays */
static void init_array(int nx, int ny,
                       DATA_TYPE POLYBENCH_2D(A, NX, NY, nx, ny),
                       DATA_TYPE POLYBENCH_1D(x, NY, ny))
{
    for (int i = 0; i < ny; i++)
        x[i] = i * M_PI;
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            A[i][j] = ((DATA_TYPE)i * (j + 1)) / nx;
}

int main(int argc, char** argv) {
    int nx = NX, ny = NY;

    /* Host arrays */
    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NX, NY, nx, ny);
    POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, NY, ny);
    POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, NY, ny);
    POLYBENCH_1D_ARRAY_DECL(tmp, DATA_TYPE, NX, nx);

    init_array(nx, ny, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(x));

    /* Host pointers */
    DATA_TYPE *A_h = &POLYBENCH_ARRAY(A)[0][0];
    DATA_TYPE *x_h = &POLYBENCH_ARRAY(x)[0];
    DATA_TYPE *y_h = &POLYBENCH_ARRAY(y)[0];
    DATA_TYPE *tmp_h = &POLYBENCH_ARRAY(tmp)[0];

    /* Sizes */
    size_t sizeA = (size_t)nx * ny * sizeof(DATA_TYPE);
    size_t sizex = ny * sizeof(DATA_TYPE);
    size_t sizey = ny * sizeof(DATA_TYPE);
    size_t sizetmp = nx * sizeof(DATA_TYPE);

    /* Device memory */
    DATA_TYPE *A_d, *x_d, *y_d, *tmp_d, *A_T_d;
    CUDA_CHECK(cudaMalloc((void**)&A_d, sizeA));
    CUDA_CHECK(cudaMalloc((void**)&x_d, sizex));
    CUDA_CHECK(cudaMalloc((void**)&y_d, sizey));
    CUDA_CHECK(cudaMalloc((void**)&tmp_d, sizetmp));
    CUDA_CHECK(cudaMalloc((void**)&A_T_d, sizeA));

    /* Copy host → device */
    CUDA_CHECK(cudaMemcpy(A_d, A_h, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(x_d, x_h, sizex, cudaMemcpyHostToDevice));

    /* Trasposizione A → A_T sulla GPU */
    dim3 blockTrans(TILE_SIZE, TILE_SIZE);
    dim3 gridTrans((ny + TILE_SIZE - 1) / TILE_SIZE, (nx + TILE_SIZE - 1) / TILE_SIZE);
    transpose_kernel<<<gridTrans, blockTrans>>>(A_d, A_T_d, nx, ny);
    CUDA_CHECK(cudaGetLastError());

    /* Timing */
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    /* Kernel tmp */
    kernel_tmp_optimized<<<nx, BLOCK_SIZE>>>(A_d, x_d, tmp_d, nx, ny);
    CUDA_CHECK(cudaGetLastError());

    /* Kernel y */
    kernel_y_optimized<<<ny, BLOCK_SIZE>>>(A_T_d, tmp_d, y_d, nx, ny);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("**************************************************\n");
    printf("GPU kernels elapsed time (OPT - TRASPOSTA): %f ms\n", ms);
    printf("**************************************************\n");

    /* Copy result back */
    CUDA_CHECK(cudaMemcpy(y_h, y_d, sizey, cudaMemcpyDeviceToHost));

    /* Cleanup */
    CUDA_CHECK(cudaFree(A_d)); CUDA_CHECK(cudaFree(x_d));
    CUDA_CHECK(cudaFree(y_d)); CUDA_CHECK(cudaFree(tmp_d));
    CUDA_CHECK(cudaFree(A_T_d));
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));

    POLYBENCH_FREE_ARRAY(A); POLYBENCH_FREE_ARRAY(x);
    POLYBENCH_FREE_ARRAY(y); POLYBENCH_FREE_ARRAY(tmp);

    return 0;
}
