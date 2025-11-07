#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is 4000. */
#include "atax.h"

/* Array initialization. */
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

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int nx,
                        DATA_TYPE POLYBENCH_1D(y, NX, nx))

{
  int i;

  for (i = 0; i < nx; i++)
  {
    fprintf(stderr, DATA_PRINTF_MODIFIER, y[i]);
    if (i % 20 == 0)
      fprintf(stderr, "\n");
  }
  fprintf(stderr, "\n");
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
/* Main computational kernel. */
static void kernel_atax(int nx, int ny,
                        DATA_TYPE POLYBENCH_2D(A, NX, NY, nx, ny),
                        DATA_TYPE POLYBENCH_1D(x, NY, ny),
                        DATA_TYPE POLYBENCH_1D(y, NY, ny),
                        DATA_TYPE POLYBENCH_1D(tmp, NX, nx))
{
  int i, j, ii, jj;
  const int BS = 512; // block size scelto per restare in cache (puoi testare 256–1024)

  // Inizializzazione
  #pragma omp parallel for schedule(static)
  for (j = 0; j < _PB_NY; j++)
    y[j] = 0.0;

  #pragma omp parallel
  {
    // Ogni thread ha un proprio buffer locale per y
    DATA_TYPE y_private[NY];
    for (int jj = 0; jj < _PB_NY; jj++)
      y_private[jj] = 0.0;

    // Loop a blocchi
    #pragma omp for collapse(2) schedule(static)
    for (ii = 0; ii < _PB_NX; ii += BS)
      for (jj = 0; jj < _PB_NY; jj += BS)
      {
        int i_max = (ii + BS < _PB_NX) ? (ii + BS) : _PB_NX;
        int j_max = (jj + BS < _PB_NY) ? (jj + BS) : _PB_NY;

        // Fase 1: tmp[i] = A[i][j] * x[j]
        for (i = ii; i < i_max; i++) {
          DATA_TYPE tmp_i = 0.0;
          #pragma omp simd
          for (j = jj; j < j_max; j++)
            tmp_i += A[i][j] * x[j];
          tmp[i] += tmp_i;
        }

        // Fase 2: y[j] += A[i][j] * tmp[i]
        for (i = ii; i < i_max; i++) {
          DATA_TYPE tmp_i = tmp[i];
          #pragma omp simd
          for (j = jj; j < j_max; j++)
            y_private[j] += A[i][j] * tmp_i;
        }
      }

    // Riduzione finale (somma i contributi locali)
    #pragma omp critical
    {
      for (int j = 0; j < _PB_NY; j++)
        y[j] += y_private[j];
    }
  } // fine parallel
}

int main(int argc, char **argv)
{
  /* Retrieve problem size. */
  int nx = NX;
  int ny = NY;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NX, NY, nx, ny);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, NY, ny);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, NY, ny);
  POLYBENCH_1D_ARRAY_DECL(tmp, DATA_TYPE, NX, nx);

  /* Initialize array(s). */
  init_array(nx, ny, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(x));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_atax(nx, ny,
              POLYBENCH_ARRAY(A),
              POLYBENCH_ARRAY(x),
              POLYBENCH_ARRAY(y),
              POLYBENCH_ARRAY(tmp));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(nx, POLYBENCH_ARRAY(y)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(y);
  POLYBENCH_FREE_ARRAY(tmp);

  return 0;
}

// Polybench nel comando per mettere il tempo
// make EXT_CFLAGS='-DMINI_DATASET -DPOLYBENCH_TIME -pg' clean all run
// gcc per non inlineare oppure non 02 ma O0 (no optimizations, piu lento)