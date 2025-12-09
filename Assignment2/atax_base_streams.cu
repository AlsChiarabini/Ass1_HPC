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
#define N_STREAMS 4

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

    int TILE = nx / N_STREAMS; // numero di RIGHE per ogni stream

    /* Crea gli stream CUDA (uno per ogni chunk) */
    cudaStream_t streams[N_STREAMS];
    for(int i = 0; i < N_STREAMS; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));  
    }

    /* Calcola dimensioni prima di allocare */
    size_t sizeA = (size_t)nx * (size_t)ny * sizeof(DATA_TYPE);
    size_t sizex = (size_t)ny * sizeof(DATA_TYPE);
    size_t sizey = (size_t)ny * sizeof(DATA_TYPE);
    size_t sizetmp = (size_t)nx * sizeof(DATA_TYPE);

    /* Con Streams siamo oblbligati ad usare pinned memory */ 
    DATA_TYPE *A_h = NULL, *x_h = NULL, *y_h = NULL, *tmp_h = NULL;
    CUDA_CHECK(cudaMallocHost((void**)&A_h, sizeA));      
    CUDA_CHECK(cudaMallocHost((void**)&x_h, sizex));      
    CUDA_CHECK(cudaMallocHost((void**)&y_h, sizey));      
    CUDA_CHECK(cudaMallocHost((void**)&tmp_h, sizetmp)); 
  
    /* Inizializza i dati nella pinned memory host */
    for (int i = 0; i < ny; i++) {
        x_h[i] = 1.0;
    }
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            //dobbiamo utilizzare un indicizzazione row-major in quanto è un puntatore singolo
            A_h[i * ny + j] = 1.0;  
        }
    }

    /* Alloca memoria DEVICE (GPU) */
    DATA_TYPE *A_d = NULL, *x_d = NULL, *y_d = NULL, *tmp_d = NULL;
    CUDA_CHECK(cudaMalloc((void**)&A_d, sizeA));
    CUDA_CHECK(cudaMalloc((void**)&x_d, sizex));
    CUDA_CHECK(cudaMalloc((void**)&y_d, sizey));
    CUDA_CHECK(cudaMalloc((void**)&tmp_d, sizetmp));

    /* il vettore x serve intero a tutti gli streams */
    CUDA_CHECK(cudaMemcpy(x_d, x_h, sizex, cudaMemcpyHostToDevice)); 

    /* y viene inizializzato a zero dato che ogni stream contribuirà */
    for (int i = 0; i < ny; i++) {
        y_h[i] = 0.0;
    }
    CUDA_CHECK(cudaMemcpy(y_d, y_h, sizey, cudaMemcpyHostToDevice));

  /* Timing: misura solo compute GPU (non include init) */
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, 0));

  /* PIPELINE: ogni stream processa un chunk di righe di A in parallelo */
  for (int s = 0; s < N_STREAMS; s++) {
      // Calcola indici per questo chunk (ogni stream = TILE righe)
      int row_start = s * TILE;
      int row_end = (s + 1) * TILE;
      int chunk_rows = row_end - row_start;
      
      // Offset in elementi (A è row-major: A[riga][col] = A[riga * ny + col])
      size_t offset_A = row_start * ny;  // Inizio del chunk nella matrice A
      size_t chunk_size_A = chunk_rows * ny * sizeof(DATA_TYPE);  // Dimensione in byte
      
      /* ASYNC: Copia chunk di A da host a device (non bloccante!) */
      CUDA_CHECK(cudaMemcpyAsync(
          &A_d[offset_A],      // Destinazione GPU: posizione del chunk
          &A_h[offset_A],      // Sorgente HOST: posizione del chunk
          chunk_size_A,        // Dimensione del chunk in byte
          cudaMemcpyHostToDevice,
          streams[s]           // Stream dedicato a questo chunk
      ));
      
      /* KERNEL su questo chunk: calcola tmp per queste righe (non bloccante!) */
      int grid_tmp = (chunk_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
      kernel_tmp<<<grid_tmp, BLOCK_SIZE, 0, streams[s]>>>(
          &A_d[offset_A],      // Puntatore al chunk di A su GPU
          x_d,                 // x completo (serve a tutti i chunk)
          &tmp_d[row_start],   // Puntatore al chunk di tmp su GPU
          chunk_rows,          // Numero di righe in questo chunk
          ny                   // Numero di colonne (sempre ny)
      );
      CUDA_CHECK(cudaGetLastError());
  }
  
  // CI ASSICURIAMO CHE TUTTI GLI STREAMS ABBIANO FINITO TMP
  for (int s = 0; s < N_STREAMS; s++) {
      CUDA_CHECK(cudaStreamSynchronize(streams[s]));
  }
  
  /* Qui usiamo A e tmp completi per calcolare y */
  int grid_y = (ny + BLOCK_SIZE - 1) / BLOCK_SIZE;
  kernel_y<<<grid_y, BLOCK_SIZE>>>(A_d, tmp_d, y_d, nx, ny);
  CUDA_CHECK(cudaGetLastError());
  
  // distruggiamo tutti gli streams
  for(int i = 0; i < N_STREAMS; i++) {
      CUDA_CHECK(cudaStreamDestroy(streams[i]));  
  }

  /* Stop timing */
  CUDA_CHECK(cudaEventRecord(stop, 0));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float milliseconds = 0;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  printf("GPU kernels elapsed time: %f ms\n", milliseconds);

  /* Copy result back PRIMA di liberare memoria */
  CUDA_CHECK(cudaMemcpy(y_h, y_d, sizey, cudaMemcpyDeviceToHost));

  /* Free memoria DEVICE (GPU) */
  CUDA_CHECK(cudaFree(A_d));
  CUDA_CHECK(cudaFree(x_d));
  CUDA_CHECK(cudaFree(y_d));
  CUDA_CHECK(cudaFree(tmp_d));

  // rilasciamo la pinned memory allocata.
  CUDA_CHECK(cudaFreeHost(A_h));  
  CUDA_CHECK(cudaFreeHost(x_h));  
  CUDA_CHECK(cudaFreeHost(y_h));  
  CUDA_CHECK(cudaFreeHost(tmp_h));

  /* Print results to prevent DCE (use existing print_array) */
  //print_array(nx, POLYBENCH_ARRAY(y));

   
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

  return 0;
}