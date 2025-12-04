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

    #if defined SEQUENTIAL
    // Inizializza l'array y a zero
    for (i = 0; i < _PB_NY; i++)
      y[i] = 0;

    // Calcola il prodotto matrice-vettore: A * x
    // tmp[i] conterrà il risultato della moltiplicazione della riga i della matrice A per il vettore x
    for (i = 0; i < _PB_NX; i++) // Ciclo sulle righe della matrice A
    {
      tmp[i] = 0;                         // Inizializza tmp[i] a zero
      for (j = 0; j < _PB_NY; j++)        // Ciclo sulle colonne della matrice A (lunghezza di x)
        tmp[i] = tmp[i] + A[i][j] * x[j]; // Somma il prodotto A[i][j] * x[j] nel vettore tmp[i]

      // Ora aggiorna il vettore y con il risultato della moltiplicazione riga di A * tmp
      for (j = 0; j < _PB_NY; j++)
        y[j] = y[j] + A[i][j] * tmp[i]; // Somma il prodotto A[i][j] * tmp[i] nel vettore y
    }

    #elif defined PARALLEL
    // Inizializza l'array y a zero
    #pragma omp parallel for
      for (int i = 0; i < _PB_NY; i++)
        y[i] = 0;

    // Calcolo parallelo di tmp[i]
    #pragma omp parallel for
      for (int i = 0; i < _PB_NX; i++)
      {
        tmp[i] = 0;
        for (int j = 0; j < _PB_NY; j++)
          tmp[i] = tmp[i] + A[i][j] * x[j];
      }

    // Aggiornamento parallelo di y[j] DA NOTARE IL CAMBIO DI VARIABILI D'ITERAZIONE
    #pragma omp parallel for
      for (int j = 0; j < _PB_NY; j++)
      {
        for (int i = 0; i < _PB_NX; i++)
          y[j] = y[j] + A[i][j] * tmp[i];
      }

    #elif defined PARALLEL_NORACE
    // Inizializza y a zero
    #pragma omp parallel for
    for (int i = 0; i < _PB_NY; i++)
      y[i] = 0;

    // Calcolo parallelo di tmp[i]
    #pragma omp parallel for
    for (int i = 0; i < _PB_NX; i++) 
    {
      tmp[i] = 0;
      for (int j = 0; j < _PB_NY; j++)
        tmp[i] += A[i][j] * x[j];
    }

    // Calcolo parallelo di y[j], evitando la race con y_private
    #pragma omp parallel
    {
      double y_private[_PB_NY];
      for (int j = 0; j < _PB_NY; j++)
        y_private[j] = 0;                 // --> diamo y_private a ogni thread per evitare race condition

      #pragma omp for
      for (int i = 0; i < _PB_NX; i++)
        for (int j = 0; j < _PB_NY; j++)
          y_private[j] += A[i][j] * tmp[i];     // ogni thread aggiorna il proprio y_private in parallelo, calcolo locale
                                                // Accesso alla matrice A in modo più efficiente (per righe)
                                                // y_private usato spesso, quindi bene per la cache

      // Riduzione manuale di y_private in y globale
      #pragma omp critical
      {
        for (int j = 0; j < _PB_NY; j++)
          y[j] += y_private[j];        // somma i privati in y globale, sequenziale (bad)
      }
    }

    #elif defined REDUCTION
    // Inizializza l'array y a zero
    #pragma omp parallel for
      for (i = 0; i < _PB_NY; i++)
        y[i] = 0;

    #pragma omp parallel for
    for (int i = 0; i < _PB_NX; i++) {
      tmp[i] = 0; // Inizializza tmp[i] a zero prima del ciclo
      for (int j = 0; j < _PB_NY; j++) {
        tmp[i] += A[i][j] * x[j]; // Accumulate directly into tmp[i]
      }
    }
    #pragma omp parallel for reduction(+:y[:_PB_NY]) // reduction esteso su array, inizializza y a 0 automaticamente
    for (int i = 0; i < _PB_NX; i++) {       // i ESTERNO (Parallelizzato)
        for (int j = 0; j < _PB_NY; j++) {   // j INTERNO (Row-Major)
            y[j] += A[i][j] * tmp[i]; 
        }
    }

    #elif defined COLLAPSE
    // Inizializzazione del vettore y
    #pragma omp parallel for
      for (int i = 0; i < ny; i++)
        y[i] = 0;

    #pragma omp parallel for collapse(2) reduction(+:tmp[:_PB_NX])
    for (int i = 0; i < _PB_NX; i++) {
        for (int j = 0; j < _PB_NY; j++) {
            tmp[i] += A[i][j] * x[j]; // Accumulate directly into tmp[i]
        }
    }
    #pragma omp parallel for collapse(2) reduction(+:y[:_PB_NY])
    for (int i = 0; i < _PB_NX; i++) { // Primo loop del collapse
        for (int j = 0; j < _PB_NY; j++) { // Secondo loop del collapse
            // L'indice combinato (i, j) è distribuito tra i thread.
            // La riduzione su y gestisce la dipendenza tra le iterazioni i
            y[j] += A[i][j] * tmp[i];
        }
    }

    #elif defined OPTIMIZED
    #pragma omp parallel for schedule(static)
    for (i = 0; i < _PB_NX; i++)
    {
        DATA_TYPE tmp_i = 0;

        #pragma omp simd reduction(+:tmp_i)
        for (j = 0; j < _PB_NY; j++)
            tmp_i += A[i][j] * x[j]; // row-major, cache-friendly

        tmp[i] = tmp_i; // Alla fine del ciclo j, tmp_i contiene il valore finale per la riga i
    }

    // Aggiornamento parallelo di y con reduction
    #pragma omp parallel for schedule(static) reduction(+:y[:_PB_NY]) // inizializza automaticamente le copie private di y a 0
    for (i = 0; i < _PB_NX; i++)
    {
        DATA_TYPE tmp_i = tmp[i];
        for (j = 0; j < _PB_NY; j++)
            y[j] += A[i][j] * tmp_i;
    }

    #elif defined OPTIMIZED_TILING
    int jj;
    const int TILE_SIZE = 256;  // dimensione blocco per cache L1/L2

    // --- Calcolo tmp[i] ---
    #pragma omp parallel for schedule(static)
    for (i = 0; i < _PB_NX; i++)
    {
        DATA_TYPE tmp_i = 0.0;

        #pragma omp simd reduction(+:tmp_i)
        for (j = 0; j < _PB_NY; j++)
            tmp_i += A[i][j] * x[j];

        tmp[i] = tmp_i;
    }

    // --- Aggiornamento y con tiling e reduction ---
    #pragma omp parallel for schedule(static) reduction(+:y[:_PB_NY])
    for (i = 0; i < _PB_NX; i++)
    {
        DATA_TYPE tmp_i = tmp[i];

        for (jj = 0; jj < _PB_NY; jj += TILE_SIZE)
        {
            int jmax = (jj + TILE_SIZE < _PB_NY) ? (jj + TILE_SIZE) : _PB_NY;

            #pragma omp simd
            for (j = jj; j < jmax; j++)
                y[j] += A[i][j] * tmp_i;
        }
    }

    #elif defined TARGET
    /*inizializza le aree di memoria da passare alla GPU passando A e x e allocando due variabili temporanee tmp e y*/
    #pragma omp target enter data map(to: A[0:_PB_NX][0:_PB_NY], x[0:_PB_NY]) map(alloc: tmp[0:_PB_NX], y[0:_PB_NY])

  //distribuisce il ciclo for sulla GPU ai vari "team" di thread
    #pragma omp target teams distribute parallel for
    for (i = 0; i < _PB_NY; i++)
      y[i] = 0;

    #pragma omp target teams distribute parallel for
    for (i = 0; i < _PB_NX; i++) {
      DATA_TYPE sum = 0;
      for (j = 0; j < _PB_NY; j++)
        sum += A[i][j] * x[j];
      tmp[i] = sum;
    }

  /*viene invertito il ciclo, da i - j a j - i per fare la trasposta
  rendendo indipendenti le somme che prima nel ciclo creavano una dipendenza*/
    #pragma omp target teams distribute parallel for
    for (j = 0; j < _PB_NY; j++) {
      DATA_TYPE sum = 0;
      for (i = 0; i < _PB_NX; i++)
        sum += A[i][j] * tmp[i];
      y[j] = sum;
    }
  //aggiorna la variabile y, il dato importante, su processore da quella su GPU
    #pragma omp target update from(y[0:_PB_NY])
  //elimina le variabili mappate sulla GPU per evitare problemi di memory leak
    #pragma omp target exit data map(delete: A[0:_PB_NX][0:_PB_NY], x[0:_PB_NY], tmp[0:_PB_NX], y[0:_PB_NY])
  
    #endif
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