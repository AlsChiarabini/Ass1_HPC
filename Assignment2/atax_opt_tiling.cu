/* atax_baseline.cu - baseline CUDA ATAX (non ottimizzato) */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

extern "C" {
    #include <polybench.h>
}
#include "atax.h"

#define BLOCK_SIZE 128
#define THREADS_PER_BLOCK 128                                                                         // fatto io

/* Simple CUDA error-check macro */
#define CUDA_CHECK(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      fprintf(stderr, "CUDA error at %s:%d -> %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(EXIT_FAILURE); \
    } \
  } while (0)

/* Kernel: tmp[i] = sum_j A[i][j] * x[j] (1 thread per riga) */
__global__ void kernel_tmp(const DATA_TYPE* A, const DATA_TYPE* x, DATA_TYPE* tmp, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nx) {
        DATA_TYPE sum = 0.0;
        for (int j = 0; j < ny; ++j) {
            sum += A[i * ny + j] * x[j];
        }
        tmp[i] = sum;
    }
}

/*
    *************************************************************
    Versione ottimizzata di kernel_tmp (spero)
    Nella versione base, avevo un thread per ogni riga, un thread faceva tutta una riga da solo, non sfruttavo i thread del blocco
    Ora, ho un blocco per ogni riga, piu thread collaborano per calcolare la somma della riga. ed uso shared per ridurre i parziali velocemente, non in global
    *************************************************************
*/ 
__global__ void kernel_tmp_optimized(const DATA_TYPE* A, const DATA_TYPE* x, DATA_TYPE* tmp, int nx, int ny) {
    // 1 blocco = 1 riga
    int row = blockIdx.x;   // ogni blocco elabora una riga, quindi id.blocco = riga
    if (row >= nx) return;

    int tx = threadIdx.x;       // Ogni thread dentro il blocco elabora un sottoinsieme della colonna di quella riga
                                // Tutti i thread del blocco collaborano per calcolare tmp[row]

    // Shared memory per somma parziale intra-block
    __shared__ DATA_TYPE sdata[THREADS_PER_BLOCK];      // alloco memoria in shared per metterci la somma parziale, piu veloce di global

    // Ogni thread calcola una porzione della riga (t0 = 0.4.8, t1 = 1.5.9...), += blockDim perche, se colonne > #thread, si scala, nell' iterazione successiva, salta di blockDim.x (128) posizioni.
    DATA_TYPE partial_sum = 0.0;
    for (int j = tx; j < ny; j += blockDim.x) { // ny = #colonne, ogni thread parte da una colonna diversa (j = tx), ogni it salta di #thread del blocco, ogni thread fa stesso lavoro
        partial_sum += A[row * ny + j] * x[j];  // A[row*ny + j] == A[row][j], con row fissa e scorro j
    }

    // Scrive la somma parziale in shared memory
    sdata[tx] = partial_sum; // ogni thread mette la sua somma parziale in questo vettore, alla fine t0 ha messo tutta la sum della colonna 0 in sdata[0]
    __syncthreads();

    // Riduzione intra-block (tree) --> e' una riduzione binaria praticamente, fatta per evitare data racing
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) { // Dopo il primo passo, gli elementi utili sono solo sdata[0..3], poi sdata[0,1] e cosi via
        if (tx < stride) {
            sdata[tx] += sdata[tx + stride];
        }
        __syncthreads();
    }
    // Alla fine di questo blocco, sdata[tx] ha gia' la somma totale

    // Thread 0 scrive il risultato finale
    if (tx == 0) {
        tmp[row] = sdata[0];    // da shared a global
    }
}
    

/* Kernel: y[j] = sum_i A[i][j] * tmp[i] (1 thread per colonna, non ottimizzato) */
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

/*
    *************************************************************
    Versione ottimizzata di kernel_y (spero)
    *************************************************************
*/ 
// kernel_y_tiled: each block handles TILE_COLS consecutive columns.
// Each thread in the block handles one column (col = blockIdx.x * TILE_COLS + threadIdx.x).
// For locality we iterate rows in chunks (ROW_TILE), so working set per iteration is small.
#define TILE_COLS 128
#define ROW_TILE 128   // chunk di righe processate nel loop interno, row_tile alla volta

__global__ void kernel_y_optimized(const DATA_TYPE* __restrict__ A, //per non far sovrapporre i puntatori
                               const DATA_TYPE* __restrict__ tmp,
                               DATA_TYPE* y,
                               int nx, int ny)
{
    // ogni blocco elabora un gruppo (tile) di colonne contigue, un thread per ogni colonna
    // Se tile_cols = 128, ogni blocco si occupa di 128 colonne, 128 threads per 128 colonne
    const int col_start = blockIdx.x * TILE_COLS; //blockId = id del blocco, che moltiplico per i thread per ogni blocco per trovare la prima colonna dello specifico blocco (quindi quella di partenza)
    const int tid = threadIdx.x; // ogni thread del blocco si occupa di una colonna diversa (da 0 a tile_cols-1)
    const int col = col_start + tid; // indice della colonna che il thread generico deve calcolare (colonna globale j)

    if (col >= ny) return; // controllo di bordo per non finire in colonne non esistenti se ny non e' multiplo di tile_cols

    DATA_TYPE sum = (DATA_TYPE)0.0;

    // Scorro le righe a blocchi di ROW_TILE, in modo che gli accessi siano coalesced, better for cache
    // For each row 'r' we load A[r*ny + col] (threads in block read contiguous columns -> coalesced).
    for (int row_base = 0; row_base < nx; row_base += ROW_TILE) {   // row_base = riga da dove iniziare, e le scorro a blocchi di ROW_TILE 
        int row_end = row_base + ROW_TILE;
        if (row_end > nx) { // semplice controllo di borso, l'ultimo tile potrebbe essere parziale
            row_end = nx;
        }

        // Inner loop: for the current small block of rows, do sequential accumulation.
        // We keep tmp[r] in a register for the multiply.
        for (int r = row_base; r < row_end; ++r) {          // in questo caso, d va da [0...127]
            // coalesced: fissato r, i thread [tid=0..(tile_cols-1)] leggono A[r*ny + col_start + tid], che sono indirizzi contigui in memoria
            sum += A[(size_t)r * (size_t)ny + (size_t)col] * tmp[r];    //A[r*ny+col] = A[r][col]. tmp[r] viene usato per tutti i thread di quella riga (viene quindi caricato in cache)
            // casto a size_t per evitare overflow per matrici di grandi dimensioni
        }
    }

    // ogni thread scrive il proprio valore in y[j] --> unica per thread, no synch
    y[col] = sum;
}                   // quindi in questo caso, t0 sta analizzando, sequenzialmente, 1 ed 4, t1 sta analizzando 2 e 5, e t2 sta analizzando 3 e 6 
                    // (ovviamente mltiplicandolo per tmp), e t0,1,2 lavorando parallelamente


/* Inizializza A e x */
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
    int nx = NX;
    int ny = NY;

    /* Allocate host arrays using PolyBench macros */
    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NX, NY, nx, ny);
    POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, NY, ny);
    POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, NY, ny);
    POLYBENCH_1D_ARRAY_DECL(tmp, DATA_TYPE, NX, nx);

    /* Initialize host arrays */
    init_array(nx, ny, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(x));

    /* Device pointers */
    DATA_TYPE *A_d = nullptr, *x_d = nullptr, *y_d = nullptr, *tmp_d = nullptr;

    size_t sizeA = (size_t)nx * (size_t)ny * sizeof(DATA_TYPE);
    size_t sizex = (size_t)ny * sizeof(DATA_TYPE);
    size_t sizey = (size_t)ny * sizeof(DATA_TYPE);
    size_t sizetmp = (size_t)nx * sizeof(DATA_TYPE);

    /* Allocate device memory */
    CUDA_CHECK(cudaMalloc((void**)&A_d, sizeA));
    CUDA_CHECK(cudaMalloc((void**)&x_d, sizex));
    CUDA_CHECK(cudaMalloc((void**)&y_d, sizey));
    CUDA_CHECK(cudaMalloc((void**)&tmp_d, sizetmp));

    /* Copy inputs to device */
    CUDA_CHECK(cudaMemcpy(A_d, &POLYBENCH_ARRAY(A)[0][0], sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(x_d, &POLYBENCH_ARRAY(x)[0], sizex, cudaMemcpyHostToDevice));

    /* GPU timing */
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));

    /*
    *************************************************************
    Versione ottimizzata di kernel_tmp (spero)
    *************************************************************
    */ 

    dim3 grid_tmp(nx); // 1 blocco per riga, la griglia ha nx blocchi
    dim3 block_tmp(THREADS_PER_BLOCK);  // threads_per_block che collaborano insieme per calcolare una riga
    kernel_tmp_optimized<<<grid_tmp, block_tmp>>>(A_d, x_d, tmp_d, nx, ny);
    CUDA_CHECK(cudaGetLastError());

    /*
    *************************************************************
    Versione ottimizzata di kernel_y (spero)
    *************************************************************
    */ 
    dim3 block_y(TILE_COLS);                                                // threads per blocco (128), 1 th = 1 colonna, quindi un blocco elabora tile_cols colonne
    dim3 grid_y( (ny + TILE_COLS - 1) / TILE_COLS );
    kernel_y_optimized<<<grid_y, block_y>>>(A_d, tmp_d, y_d, nx, ny);
    CUDA_CHECK(cudaGetLastError());


    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("**************************************************\n");
    printf("GPU kernels elapsed time (OPT - TILING): %f ms\n", milliseconds);
    printf("**************************************************\n");

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(&POLYBENCH_ARRAY(y)[0], y_d, sizey, cudaMemcpyDeviceToHost));

    /* Optional: print for verification */
    // print_array(ny, POLYBENCH_ARRAY(y));

    /* Free device memory */
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(x_d));
    CUDA_CHECK(cudaFree(y_d));
    CUDA_CHECK(cudaFree(tmp_d));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    /* Free host arrays */
    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(x);
    POLYBENCH_FREE_ARRAY(y);
    POLYBENCH_FREE_ARRAY(tmp);

    return 0;
}
