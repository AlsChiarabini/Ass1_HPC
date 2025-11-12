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
static void kernel_atax(int nx, int ny,
                        DATA_TYPE POLYBENCH_2D(A, NX, NY, nx, ny),
                        DATA_TYPE POLYBENCH_1D(x, NY, ny),
                        DATA_TYPE POLYBENCH_1D(y, NY, ny),
                        DATA_TYPE POLYBENCH_1D(tmp, NX, nx))
{
  int i, j;
/*inizializza le aree di memoria da passare alla GPU passando A e x e
allocando due variabili temporanee tmp e y*/
  #pragma omp target enter data map(to: A[0:nx][0:ny], x[0:ny]) map(alloc: tmp[0:nx], y[0:ny])

//distribuisce il ciclo for sulla GPU ai vari "team" di thread
  #pragma omp target teams distribute parallel for
  for (i = 0; i < ny; i++)
    y[i] = 0;

  #pragma omp target teams distribute parallel for
  for (i = 0; i < nx; i++) {
    DATA_TYPE sum = 0;
    for (j = 0; j < ny; j++)
      sum += A[i][j] * x[j];
    tmp[i] = sum;
  }

/*viene invertito il ciclo, da i - j a j - i per fare la trasposta
rendendo indipendenti le somme che prima nel ciclo creavano una dipendenza*/
  #pragma omp target teams distribute parallel for
  for (j = 0; j < ny; j++) {
    DATA_TYPE sum = 0;
    for (i = 0; i < nx; i++)
      sum += A[i][j] * tmp[i];
    y[j] = sum;
  }
//aggiorna la variabile y, il dato importante, su processore da quella su GPU
  #pragma omp target update from(y[0:ny])
//elimina le variabili mappate sulla GPU per evitare problemi di memory leak
  #pragma omp target exit data map(delete: A[0:nx][0:ny], x[0:ny], tmp[0:nx], y[0:ny])
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
