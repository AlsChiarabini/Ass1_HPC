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
__global__ void kernel_tmp(const DATA_TYPE* A, const DATA_TYPE* x, DATA_TYPE* tmp, int nx, int ny)
{
    // Calcolo la riga che questo warp/blocco deve gestire
    int row = blockIdx.x;   // un blocco per riga
    if (row >= nx) return;

    // Thread all'interno del blocco
    int tid = threadIdx.x;
    int blockDimX = blockDim.x; // dimensione del blocco

    __shared__ DATA_TYPE tileX[BLOCK_SIZE];
    __shared__ DATA_TYPE partialSum[BLOCK_SIZE]; // accumulo parziale per ogni thread

    DATA_TYPE sum = 0.0f;

    // Ciclo sui tile di colonne
    for (int t = 0; t < ny; t += BLOCK_SIZE)
    {
        int tileLimit = min(BLOCK_SIZE, ny - t);

        // Caricamento tile di x in shared memory
        if (tid < tileLimit)
            tileX[tid] = x[t + tid];

        __syncthreads();

        // Ogni thread calcola una parte del tile
        partialSum[tid] = 0.0f;
        for (int k = tid; k < tileLimit; k += blockDimX)
        {
            partialSum[tid] += A[row * ny + (t + k)] * tileX[k];
        }

        __syncthreads();

        // Riduzione in shared memory per sommare i risultati parziali
        for (int stride = blockDimX / 2; stride > 0; stride /= 2)
        {
            if (tid < stride && tid + stride < tileLimit)
                partialSum[tid] += partialSum[tid + stride];
            __syncthreads();
        }

        if (tid == 0)
            sum += partialSum[0];

        __syncthreads(); // sicuro per il prossimo tile
    }

    // Scrittura del risultato finale
    if (tid == 0)
        tmp[row] = sum;
}


/* Kernel: compute y[j] = sum_i A[i][j] * tmp[i]  (one thread per column j) */
__global__ void kernel_y(const DATA_TYPE* A_T, const DATA_TYPE* tmp, DATA_TYPE* y, int nx, int ny)
{
    int col = blockIdx.x;          // un blocco per colonna j
    if (col >= ny) return;

    int tid = threadIdx.x;
    int blockDimX = blockDim.x;

    __shared__ DATA_TYPE partialSum[BLOCK_SIZE]; // somma parziale per ogni thread

    DATA_TYPE sum = 0.0f;

    // Ciclo sui tile di righe
    for (int t = 0; t < nx; t += BLOCK_SIZE)
    {
        int tileLimit = min(BLOCK_SIZE, nx - t);

        // Ogni thread calcola una parte del tile
        partialSum[tid] = 0.0f;
        for (int i = tid; i < tileLimit; i += blockDimX)
        {
            partialSum[tid] += A_T[col * nx + (t + i)] * tmp[t + i];
        }

        __syncthreads();

        // Riduzione in shared memory
        for (int stride = blockDimX / 2; stride > 0; stride /= 2)
        {
            if (tid < stride && tid + stride < tileLimit)
                partialSum[tid] += partialSum[tid + stride];
            __syncthreads();
        }

        if (tid == 0)
            sum += partialSum[0];

        __syncthreads(); // sicuro per il prossimo tile
    }

    if (tid == 0)
        y[col] = sum;
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
  init_array(nx, ny, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(x)); // qua inizializzo gli array A e x

  /* Pointers to host raw memory (row-major) */
  DATA_TYPE* A_h = &POLYBENCH_ARRAY(A)[0][0]; // alloco matrice A in memoria continua
  DATA_TYPE* x_h = &POLYBENCH_ARRAY(x)[0];    // alloco vettore x in memoria continua
  DATA_TYPE* y_h = &POLYBENCH_ARRAY(y)[0];    // alloco vettore y in memoria continua
  DATA_TYPE* tmp_h = &POLYBENCH_ARRAY(tmp)[0]; // alloco vettore tmp in memoria continua

  /* Device pointers */
  DATA_TYPE *A_d = NULL, *x_d = NULL, *y_d = NULL, *tmp_d = NULL; // puntatori in device

  size_t sizeA = (size_t)nx * (size_t)ny * sizeof(DATA_TYPE); // dimensione matrice A
  size_t sizex = (size_t)ny * sizeof(DATA_TYPE);            // dimensione vettore x 
  size_t sizey = (size_t)ny * sizeof(DATA_TYPE);            // dimensione vettore y
  size_t sizetmp = (size_t)nx * sizeof(DATA_TYPE);          // dimensione vettore tmp

  /* Allocate device memory */
  CUDA_CHECK(cudaMalloc((void**)&A_d, sizeA));    // alloco matrice A in device --> vedi che sono puntatori a puntatori
  CUDA_CHECK(cudaMalloc((void**)&x_d, sizex));    // alloco vettore x in device
  CUDA_CHECK(cudaMalloc((void**)&y_d, sizey));    // alloco vettore y in device
  CUDA_CHECK(cudaMalloc((void**)&tmp_d, sizetmp)); // alloco vettore tmp in device
  // ***************************************************************************************
  DATA_TYPE *A_T_d = NULL;
  CUDA_CHECK(cudaMalloc((void**)&A_T_d, sizeA));
  // Trasponi A_h in host
    DATA_TYPE* A_T_h = (DATA_TYPE*)malloc(sizeA);
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            A_T_h[j * nx + i] = A_h[i * ny + j];
    // ***************************************************************************************

// Copia A_T in device
CUDA_CHECK(cudaMemcpy(A_T_d, A_T_h, sizeA, cudaMemcpyHostToDevice));
free(A_T_h);



  /* Copy inputs to device */
  CUDA_CHECK(cudaMemcpy(A_d, A_h, sizeA, cudaMemcpyHostToDevice)); // copio solo A ed x, y e tmp li calcolo nei kernel
  CUDA_CHECK(cudaMemcpy(x_d, x_h, sizex, cudaMemcpyHostToDevice));  // copio vettore x in device
  /* No need to init y_d or tmp_d (kernels write them) */

  /* Launch kernels and measure time with cudaEvent */
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, 0));

  /* Compute tmp */
  int block = BLOCK_SIZE;                                     // numero di thread per blocco
  int grid_tmp = (nx + block - 1) / block;                   // calcolo il numero di blocchi necessari
  kernel_tmp<<<grid_tmp, block>>>(A_d, x_d, tmp_d, nx, ny); // lancio kernel per calcolare tmp (il primo passo)
  CUDA_CHECK(cudaGetLastError());                               // --> ogni thread calcola una riga di tmp

  /* Compute y */
  int grid_y = (ny + block - 1) / block;                 // calcolo il numero di blocchi necessari
  // **************************************************************************************
  kernel_y<<<grid_y, block>>>(A_T_d, tmp_d, y_d, nx, ny); // lancio kernel per calcolare y (il secondo passo)
  CUDA_CHECK(cudaGetLastError());                           // --> ogni thread calcola una colonna di y (COLONNA, no coalescence!!!!)

  /* Synchronize and stop timer */
  CUDA_CHECK(cudaDeviceSynchronize());        // ensure all kernels are done
  CUDA_CHECK(cudaEventRecord(stop, 0));       // record the stop event
  CUDA_CHECK(cudaEventSynchronize(stop));

  float milliseconds = 0;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  printf("GPU kernels elapsed time: %f ms\n", milliseconds);

  /* Copy result back */
  CUDA_CHECK(cudaMemcpy(y_h, y_d, sizey, cudaMemcpyDeviceToHost));  // sul device ho calcolato y, lo copio in y_h in host

  /* Print results to prevent DCE (use existing print_array) */
  //print_array(nx, POLYBENCH_ARRAY(y));

  /* Free device memory */
  CUDA_CHECK(cudaFree(A_d));
  CUDA_CHECK(cudaFree(x_d));
  CUDA_CHECK(cudaFree(y_d));
  CUDA_CHECK(cudaFree(tmp_d));
  // ****************************************************** prova
  CUDA_CHECK(cudaFree(A_T_d));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  /* Free host arrays (polybench macros) */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(y);
  POLYBENCH_FREE_ARRAY(tmp);

  return 0;
}

//Per compilare (devo ancora provare) make EXT_CXXFLAGS='-DLARGE_DATASET -DPOLYBENCH_TIME -pg' clean all run
