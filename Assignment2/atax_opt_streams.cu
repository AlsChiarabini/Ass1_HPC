/* atax_opt_streams.cu - Combina STREAMS + TILING ottimizzato */

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

/* CUDA error-check macro */
#define CUDA_CHECK(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      fprintf(stderr, "CUDA error at %s:%d -> %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(EXIT_FAILURE); \
    } \
  } while (0)

/* Kernel ottimizzato tmp: usa shared memory + riduzione tree */
__global__ void kernel_tmp_optimized(const DATA_TYPE* A, const DATA_TYPE* x, DATA_TYPE* tmp, int nx, int ny) {
    int row = blockIdx.x;   // 1 blocco = 1 riga
    if (row >= nx) return;

    int tx = threadIdx.x;
    __shared__ DATA_TYPE sdata[THREADS_PER_BLOCK];

    // Ogni thread calcola somma parziale della riga
    DATA_TYPE partial_sum = 0.0;
    for (int j = tx; j < ny; j += blockDim.x) {
        partial_sum += A[row * ny + j] * x[j];
    }

    // Scrive in shared memory
    sdata[tx] = partial_sum;
    __syncthreads();

    // Riduzione tree intra-block
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tx < stride) {
            sdata[tx] += sdata[tx + stride];
        }
        __syncthreads();
    }

    // Thread 0 scrive risultato finale
    if (tx == 0) {
        tmp[row] = sdata[0];
    }
}

/* Kernel ottimizzato y: tiling per località cache */
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

    // Itera su righe in chunk per località cache
    for (int row_base = 0; row_base < nx; row_base += ROW_TILE) {
        int row_end = row_base + ROW_TILE;
        if (row_end > nx) row_end = nx;

        for (int r = row_base; r < row_end; ++r) {
            sum += A[(size_t)r * (size_t)ny + (size_t)col] * tmp[r];
        }
    }

    y[col] = sum;
}

int main(int argc, char** argv) {
    int nx = NX;
    int ny = NY;

    int TILE = nx / N_STREAMS; // righe per stream

    /* Crea streams CUDA */
    cudaStream_t streams[N_STREAMS];
    for(int i = 0; i < N_STREAMS; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));  
    }

    /* Calcola dimensioni */
    size_t sizeA = (size_t)nx * (size_t)ny * sizeof(DATA_TYPE);
    size_t sizex = (size_t)ny * sizeof(DATA_TYPE);
    size_t sizey = (size_t)ny * sizeof(DATA_TYPE);
    size_t sizetmp = (size_t)nx * sizeof(DATA_TYPE);

    /* Alloca pinned HOST (per cudaMemcpyAsync) */
    DATA_TYPE *A_h = NULL, *x_h = NULL, *y_h = NULL, *tmp_h = NULL;
    CUDA_CHECK(cudaMallocHost((void**)&A_h, sizeA));      
    CUDA_CHECK(cudaMallocHost((void**)&x_h, sizex));      
    CUDA_CHECK(cudaMallocHost((void**)&y_h, sizey));      
    CUDA_CHECK(cudaMallocHost((void**)&tmp_h, sizetmp)); 
  
    /* Inizializza dati */
    for (int i = 0; i < ny; i++)
        x_h[i] = 1.0;
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            A_h[i * ny + j] = 1.0;

    /* Alloca GPU memory */
    DATA_TYPE *A_d = NULL, *x_d = NULL, *y_d = NULL, *tmp_d = NULL;
    CUDA_CHECK(cudaMalloc((void**)&A_d, sizeA));
    CUDA_CHECK(cudaMalloc((void**)&x_d, sizex));
    CUDA_CHECK(cudaMalloc((void**)&y_d, sizey));
    CUDA_CHECK(cudaMalloc((void**)&tmp_d, sizetmp));

    /* Copia x (serve intero a tutti gli streams) */
    CUDA_CHECK(cudaMemcpy(x_d, x_h, sizex, cudaMemcpyHostToDevice)); 

    /* Inizializza y a zero */
    for (int i = 0; i < ny; i++)
        y_h[i] = 0.0;
    CUDA_CHECK(cudaMemcpy(y_d, y_h, sizey, cudaMemcpyHostToDevice));

    /* Timing */
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));

    /* PIPELINE STREAMS: copia chunk + kernel_tmp ottimizzato per chunk */
    for (int s = 0; s < N_STREAMS; s++) {
        int row_start = s * TILE;
        int row_end = (s + 1) * TILE;
        int chunk_rows = row_end - row_start;
        
        size_t offset_A = row_start * ny;
        size_t chunk_size_A = chunk_rows * ny * sizeof(DATA_TYPE);
        
        /* Copia chunk async */
        CUDA_CHECK(cudaMemcpyAsync(
            &A_d[offset_A],
            &A_h[offset_A],
            chunk_size_A,
            cudaMemcpyHostToDevice,
            streams[s]
        ));
        
        /* Kernel_tmp ottimizzato su chunk (1 blocco per riga) */
        kernel_tmp_optimized<<<chunk_rows, THREADS_PER_BLOCK, 0, streams[s]>>>(
            &A_d[offset_A],
            x_d,
            &tmp_d[row_start],
            chunk_rows,
            ny
        );
        CUDA_CHECK(cudaGetLastError());
    }
    
    /* Sincronizza tutti gli stream (tmp completo necessario per kernel_y) */
    for (int s = 0; s < N_STREAMS; s++) {
        CUDA_CHECK(cudaStreamSynchronize(streams[s]));
    }
    
    /* Kernel_y ottimizzato (usa A e tmp completi) */
    int grid_y = (ny + TILE_COLS - 1) / TILE_COLS;
    kernel_y_optimized<<<grid_y, TILE_COLS>>>(A_d, tmp_d, y_d, nx, ny);
    CUDA_CHECK(cudaGetLastError());
    
    /* Distruggi streams */
    for(int i = 0; i < N_STREAMS; i++) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));  
    }

    /* Stop timing */
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("GPU kernels elapsed time: %f ms\n", milliseconds);

    /* Copia risultato */
    CUDA_CHECK(cudaMemcpy(y_h, y_d, sizey, cudaMemcpyDeviceToHost));

    /* Free GPU */
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(x_d));
    CUDA_CHECK(cudaFree(y_d));
    CUDA_CHECK(cudaFree(tmp_d));

    /* Free pinned HOST */
    CUDA_CHECK(cudaFreeHost(A_h));  
    CUDA_CHECK(cudaFreeHost(x_h));  
    CUDA_CHECK(cudaFreeHost(y_h));  
    CUDA_CHECK(cudaFreeHost(tmp_h)); 

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
