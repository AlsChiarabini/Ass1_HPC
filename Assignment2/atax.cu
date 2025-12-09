/* atax.cu - Versione CUDA unificata con tutte le ottimizzazioni tramite macro
   Compilare con: make EXERCISE=atax.cu EXT_CFLAGS='-DBASE -DLARGE_DATASET' clean all run
   Macro disponibili: BASE, PINNED, UVM, CONST, TILING, STREAMS, OPT_STREAMS
*/

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
#define N_STREAMS 4
#define TILE_SIZE 16

/* CUDA error-check macro */
#define CUDA_CHECK(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      fprintf(stderr, "CUDA error at %s:%d -> %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(EXIT_FAILURE); \
    } \
  } while (0)

/* ========== CONSTANT MEMORY (solo per versione CONST) ========== */
#ifdef CONST
__constant__ DATA_TYPE x_d_const[8192];
#endif

/* ========== KERNEL BASE ========== */
__global__ void kernel_tmp(const DATA_TYPE* A, const DATA_TYPE* x, DATA_TYPE* tmp, int nx, int ny) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nx) {
    DATA_TYPE sum = 0.0;
    int row_off = i * ny;
    for (int j = 0; j < ny; ++j) {
      #ifdef CONST
        sum += A[row_off + j] * x_d_const[j];
      #else
        sum += A[row_off + j] * x[j];
      #endif
    }
    tmp[i] = sum;
  }
}

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

/* ========== KERNEL OTTIMIZZATI (TILING e OPT_STREAMS) ========== */
#if defined(TILING) || defined(OPT_STREAMS)

__global__ void kernel_tmp_optimized(const DATA_TYPE* A, const DATA_TYPE* x, DATA_TYPE* tmp, int nx, int ny) {
    int row = blockIdx.x;
    if (row >= nx) return;

    int tx = threadIdx.x;
    __shared__ DATA_TYPE sdata[THREADS_PER_BLOCK];

    DATA_TYPE partial_sum = 0.0;
    for (int j = tx; j < ny; j += blockDim.x) {
        partial_sum += A[row * ny + j] * x[j];
    }

    sdata[tx] = partial_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tx < stride) {
            sdata[tx] += sdata[tx + stride];
        }
        __syncthreads();
    }

    if (tx == 0) {
        tmp[row] = sdata[0];
    }
}

#define TILE_COLS 128
#define ROW_TILE 128

__global__ void kernel_y_optimized(const DATA_TYPE* __restrict__ A,
                                   const DATA_TYPE* __restrict__ tmp,
                                   DATA_TYPE* y,
                                   int nx, int ny)
{
    const int col_start = blockIdx.x * TILE_COLS;
    const int tid = threadIdx.x;
    const int col = col_start + tid;

    if (col >= ny) return;

    DATA_TYPE sum = 0.0;

    for (int row_base = 0; row_base < nx; row_base += ROW_TILE) {
        int row_end = row_base + ROW_TILE;
        if (row_end > nx) row_end = nx;

        for (int r = row_base; r < row_end; ++r) {
            sum += A[(size_t)r * (size_t)ny + (size_t)col] * tmp[r];
        }
    }

    y[col] = sum;
}

#endif

/* ========== KERNEL TRANSPOSE ========== */
#ifdef TRANSPOSE

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

__global__ void kernel_tmp_optimized_transpose(const DATA_TYPE* A, const DATA_TYPE* x, DATA_TYPE* tmp, int nx, int ny) {
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

__global__ void kernel_y_optimized_transpose(const DATA_TYPE* A_T, const DATA_TYPE* tmp, DATA_TYPE* y, int nx, int ny) {
    int col = blockIdx.x;
    if (col >= ny) return;

    int tid = threadIdx.x;
    __shared__ DATA_TYPE sdata[BLOCK_SIZE];

    DATA_TYPE sum = 0.0;
    for (int t = 0; t < nx; t += BLOCK_SIZE) {
        int tileLimit = (t + BLOCK_SIZE < nx) ? BLOCK_SIZE : (nx - t);
        sdata[tid] = 0.0;

        for (int i = tid; i < tileLimit; i += blockDim.x)
            sdata[tid] += A_T[col * nx + (t + i)] * tmp[t + i];

        __syncthreads();

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

#endif

/* ========== MAIN ========== */
int main(int argc, char** argv) {
    int nx = NX;
    int ny = NY;

    size_t sizeA = (size_t)nx * (size_t)ny * sizeof(DATA_TYPE);
    size_t sizex = (size_t)ny * sizeof(DATA_TYPE);
    size_t sizey = (size_t)ny * sizeof(DATA_TYPE);
    size_t sizetmp = (size_t)nx * sizeof(DATA_TYPE);

    DATA_TYPE *A_h = NULL, *x_h = NULL, *y_h = NULL, *tmp_h = NULL;
    DATA_TYPE *A_d = NULL, *x_d = NULL, *y_d = NULL, *tmp_d = NULL;
    #ifdef TRANSPOSE
    DATA_TYPE *A_T_d = NULL;
    #endif

    /* ========== ALLOCAZIONE MEMORIA ========== */
    #if defined(UVM)
        /* Unified Memory */
        CUDA_CHECK(cudaMallocManaged(&A_h, sizeA));
        CUDA_CHECK(cudaMallocManaged(&x_h, sizex));
        CUDA_CHECK(cudaMallocManaged(&y_h, sizey));
        CUDA_CHECK(cudaMallocManaged(&tmp_h, sizetmp));
        A_d = A_h; x_d = x_h; y_d = y_h; tmp_d = tmp_h;

    #elif defined(PINNED) || defined(STREAMS) || defined(OPT_STREAMS)
        /* Pinned Host + Device */
        CUDA_CHECK(cudaMallocHost((void**)&A_h, sizeA));
        CUDA_CHECK(cudaMallocHost((void**)&x_h, sizex));
        CUDA_CHECK(cudaMallocHost((void**)&y_h, sizey));
        CUDA_CHECK(cudaMallocHost((void**)&tmp_h, sizetmp));

        CUDA_CHECK(cudaMalloc((void**)&A_d, sizeA));
        CUDA_CHECK(cudaMalloc((void**)&x_d, sizex));
        CUDA_CHECK(cudaMalloc((void**)&y_d, sizey));
        CUDA_CHECK(cudaMalloc((void**)&tmp_d, sizetmp));

    #else
        /* BASE, CONST, TILING: PolyBench + Device */
        POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NX, NY, nx, ny);
        POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, NY, ny);
        POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, NY, ny);
        POLYBENCH_1D_ARRAY_DECL(tmp, DATA_TYPE, NX, nx);

        A_h = &POLYBENCH_ARRAY(A)[0][0];
        x_h = &POLYBENCH_ARRAY(x)[0];
        y_h = &POLYBENCH_ARRAY(y)[0];
        tmp_h = &POLYBENCH_ARRAY(tmp)[0];

        CUDA_CHECK(cudaMalloc((void**)&A_d, sizeA));
        CUDA_CHECK(cudaMalloc((void**)&x_d, sizex));
        CUDA_CHECK(cudaMalloc((void**)&y_d, sizey));
        CUDA_CHECK(cudaMalloc((void**)&tmp_d, sizetmp));
        #ifdef TRANSPOSE
        CUDA_CHECK(cudaMalloc((void**)&A_T_d, sizeA));
        #endif
    #endif

    /* ========== INIZIALIZZAZIONE ========== */
    #if defined(PINNED) || defined(STREAMS) || defined(OPT_STREAMS)
        for (int i = 0; i < ny; i++)
            x_h[i] = i * M_PI;
        for (int i = 0; i < nx; i++)
            for (int j = 0; j < ny; j++)
                A_h[i * ny + j] = ((DATA_TYPE)i * (j + 1)) / nx;
    #else
        for (int i = 0; i < ny; i++)
            x_h[i] = i * M_PI;
        for (int i = 0; i < nx; i++)
            for (int j = 0; j < ny; j++)
                A_h[i * ny + j] = ((DATA_TYPE)i * (j + 1)) / nx;
    #endif

    /* ========== COPIA HOST->DEVICE ========== */
    #if !defined(UVM)
        #if defined(STREAMS) || defined(OPT_STREAMS)
            /* Per streams, copia solo x ora (A viene copiato a chunk) */
            CUDA_CHECK(cudaMemcpy(x_d, x_h, sizex, cudaMemcpyHostToDevice));
            for (int i = 0; i < ny; i++) y_h[i] = 0.0;
            CUDA_CHECK(cudaMemcpy(y_d, y_h, sizey, cudaMemcpyHostToDevice));
        #else
            CUDA_CHECK(cudaMemcpy(A_d, A_h, sizeA, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(x_d, x_h, sizex, cudaMemcpyHostToDevice));
            #ifdef CONST
                if (ny > 8192) {
                    fprintf(stderr, "Error: ny=%d exceeds constant memory limit (8192)\n", ny);
                    exit(1);
                }
                CUDA_CHECK(cudaMemcpyToSymbol(x_d_const, x_h, sizex));
            #endif
        #endif
    #endif

    /* ========== TIMING ========== */
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));

    /* ========== ESECUZIONE KERNEL ========== */
    #if defined(STREAMS) || defined(OPT_STREAMS)
        /* Versione con STREAMS */
        cudaStream_t streams[N_STREAMS];
        for(int i = 0; i < N_STREAMS; i++) {
            CUDA_CHECK(cudaStreamCreate(&streams[i]));
        }

        int TILE = nx / N_STREAMS;
        for (int s = 0; s < N_STREAMS; s++) {
            int row_start = s * TILE;
            int row_end = (s + 1) * TILE;
            int chunk_rows = row_end - row_start;
            
            size_t offset_A = row_start * ny;
            size_t chunk_size_A = chunk_rows * ny * sizeof(DATA_TYPE);
            
            CUDA_CHECK(cudaMemcpyAsync(
                &A_d[offset_A],
                &A_h[offset_A],
                chunk_size_A,
                cudaMemcpyHostToDevice,
                streams[s]
            ));
            
            #ifdef OPT_STREAMS
                kernel_tmp_optimized<<<chunk_rows, THREADS_PER_BLOCK, 0, streams[s]>>>(
                    &A_d[offset_A], x_d, &tmp_d[row_start], chunk_rows, ny
                );
            #else
                int grid_tmp = (chunk_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
                kernel_tmp<<<grid_tmp, BLOCK_SIZE, 0, streams[s]>>>(
                    &A_d[offset_A], x_d, &tmp_d[row_start], chunk_rows, ny
                );
            #endif
            CUDA_CHECK(cudaGetLastError());
        }

        for (int s = 0; s < N_STREAMS; s++) {
            CUDA_CHECK(cudaStreamSynchronize(streams[s]));
        }

        #ifdef OPT_STREAMS
            int grid_y = (ny + TILE_COLS - 1) / TILE_COLS;
            kernel_y_optimized<<<grid_y, TILE_COLS>>>(A_d, tmp_d, y_d, nx, ny);
        #else
            int grid_y = (ny + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernel_y<<<grid_y, BLOCK_SIZE>>>(A_d, tmp_d, y_d, nx, ny);
        #endif
        CUDA_CHECK(cudaGetLastError());

        for(int i = 0; i < N_STREAMS; i++) {
            CUDA_CHECK(cudaStreamDestroy(streams[i]));
        }

    #elif defined(TRANSPOSE)
        /* Versione TRANSPOSE: trasponi A prima di usarlo */
        dim3 blockTrans(TILE_SIZE, TILE_SIZE);
        dim3 gridTrans((ny + TILE_SIZE - 1) / TILE_SIZE, (nx + TILE_SIZE - 1) / TILE_SIZE);
        transpose_kernel<<<gridTrans, blockTrans>>>(A_d, A_T_d, nx, ny);
        CUDA_CHECK(cudaGetLastError());

        kernel_tmp_optimized_transpose<<<nx, BLOCK_SIZE>>>(A_d, x_d, tmp_d, nx, ny);
        CUDA_CHECK(cudaGetLastError());

        kernel_y_optimized_transpose<<<ny, BLOCK_SIZE>>>(A_T_d, tmp_d, y_d, nx, ny);
        CUDA_CHECK(cudaGetLastError());

    #elif defined(TILING)
        /* Versione TILING ottimizzato */
        kernel_tmp_optimized<<<nx, THREADS_PER_BLOCK>>>(A_d, x_d, tmp_d, nx, ny);
        CUDA_CHECK(cudaGetLastError());

        int grid_y = (ny + TILE_COLS - 1) / TILE_COLS;
        kernel_y_optimized<<<grid_y, TILE_COLS>>>(A_d, tmp_d, y_d, nx, ny);
        CUDA_CHECK(cudaGetLastError());

    #else
        /* BASE, PINNED, UVM, CONST */
        int grid_tmp = (nx + BLOCK_SIZE - 1) / BLOCK_SIZE;
        #ifdef CONST
            kernel_tmp<<<grid_tmp, BLOCK_SIZE>>>(A_d, NULL, tmp_d, nx, ny);
        #else
            kernel_tmp<<<grid_tmp, BLOCK_SIZE>>>(A_d, x_d, tmp_d, nx, ny);
        #endif
        CUDA_CHECK(cudaGetLastError());

        int grid_y = (ny + BLOCK_SIZE - 1) / BLOCK_SIZE;
        kernel_y<<<grid_y, BLOCK_SIZE>>>(A_d, tmp_d, y_d, nx, ny);
        CUDA_CHECK(cudaGetLastError());
    #endif

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("GPU kernels elapsed time: %f ms\n", milliseconds);

    /* ========== COPIA DEVICE->HOST ========== */
    #if !defined(UVM)
        CUDA_CHECK(cudaMemcpy(y_h, y_d, sizey, cudaMemcpyDeviceToHost));
    #endif

    /* ========== CLEANUP ========== */
    #if defined(UVM)
        CUDA_CHECK(cudaFree(A_h));
        CUDA_CHECK(cudaFree(x_h));
        CUDA_CHECK(cudaFree(y_h));
        CUDA_CHECK(cudaFree(tmp_h));
    #elif defined(PINNED) || defined(STREAMS) || defined(OPT_STREAMS)
        CUDA_CHECK(cudaFree(A_d));
        CUDA_CHECK(cudaFree(x_d));
        CUDA_CHECK(cudaFree(y_d));
        CUDA_CHECK(cudaFree(tmp_d));
        CUDA_CHECK(cudaFreeHost(A_h));
        CUDA_CHECK(cudaFreeHost(x_h));
        CUDA_CHECK(cudaFreeHost(y_h));
        CUDA_CHECK(cudaFreeHost(tmp_h));
    #else
        CUDA_CHECK(cudaFree(A_d));
        CUDA_CHECK(cudaFree(x_d));
        CUDA_CHECK(cudaFree(y_d));
        CUDA_CHECK(cudaFree(tmp_d));
        #ifdef TRANSPOSE
        CUDA_CHECK(cudaFree(A_T_d));
        #endif
        POLYBENCH_FREE_ARRAY(A);
        POLYBENCH_FREE_ARRAY(x);
        POLYBENCH_FREE_ARRAY(y);
        POLYBENCH_FREE_ARRAY(tmp);
    #endif

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
