/* POLYBENCH/GPU-OPENMP
 *
 * This file is a part of the Polybench/GPU-OpenMP suite
 *
 * Contact:
 * William Killian <killian@udel.edu>
 * 
 * Copyright 2013, The University of Delaware
 */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include "../util/timer.h"
#include "../util/bbop_manager.h"

/* Include polybench common header. */
#include "polybench.h"
#include <stdint.h>


/* Include benchmark-specific header. */
/* Default data type is double, default size is 4000. */
#include "gemm.h"


/* Array initialization. */
static
void init_array(int ni, int nj, int nk,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj),
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj))
{
  int i, j;

  *alpha = 32412;
  *beta = 2123;
  for (i = 0; i < ni; i++){
    for (j = 0; j < nj; j++){
      DATA_TYPE tmp = ((DATA_TYPE) i*j) / ni;

      if (INT4_DATA_TYPE == 1){
        tmp = tmp % 16;
      }

      C[i][j] = tmp;
    }
  }

  for (i = 0; i < ni; i++){
    for (j = 0; j < nk; j++){
      DATA_TYPE tmp = ((DATA_TYPE) i*j) / ni;

      if (INT4_DATA_TYPE == 1){
        tmp = tmp % 16;
      }
      A[i][j] = tmp;
    }
  }

  for (i = 0; i < nk; i++){
    for (j = 0; j < nj; j++){
      DATA_TYPE tmp = ((DATA_TYPE) i*j) / ni;

      if (INT4_DATA_TYPE == 1){
        tmp = tmp % 16;
      }
      B[i][j] = tmp;   
    }
  }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, int nj,
		 DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj))
{
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
	fprintf (stderr, DATA_PRINTF_MODIFIER, C[i][j]);
	if ((i * ni + j) % 20 == 0) fprintf (stderr, "\n");
    }
  fprintf (stderr, "\n");
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_gemm(int ni, int nj, int nk,
		 DATA_TYPE alpha,
		 DATA_TYPE beta,
		 DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj),
		 DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		 DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj))
{
  int i, j, k;

  #ifdef RUN_BASELINE
    printf("Running baseline\n");
  
    #pragma scop
    #pragma omp parallel
    {
      /* C := alpha*A*B + beta*C */
      #pragma omp for private (j, k)
      for (i = 0; i < _PB_NI; i++)
        for (j = 0; j < _PB_NJ; j++){
          C[i][j] *= beta;
          
          for (k = 0; k < _PB_NK; ++k)
            C[i][j] += alpha * A[i][k] * B[k][j];
        }
    }
    #pragma endscop
  #endif

  #ifdef RUN_PIM
    printf("Running PIM version\n");

    // Create a temporary array for beta
    DATA_TYPE array_beta[_PB_NJ];
    DATA_TYPE array_alpha[_PB_NK];

    for (i = 0; i < _PB_NJ; i++){
      array_beta[i] = beta;
      if (INT4_DATA_TYPE == 1){
          array_beta[i] = (array_beta[i] > INT4_MAX) ? INT4_MAX : array_beta[i];
          array_beta[i] = (array_beta[i] < INT4_MIN) ? INT4_MIN : array_beta[i];
      }
    }

    for (i = 0; i < _PB_NK; i++){
      array_alpha[i] = alpha;
      if (INT4_DATA_TYPE == 1){
          array_alpha[i] = (array_alpha[i] > INT4_MAX) ? INT4_MAX : array_alpha[i];
          array_alpha[i] = (array_alpha[i] < INT4_MIN) ? INT4_MIN : array_alpha[i];
      }
    }

    DATA_TYPE output_a_b[_PB_NK];
    DATA_TYPE output_a_b_alpha[_PB_NK];
    DATA_TYPE transposed_b[_PB_NK];

    /* C := alpha*A*B + beta*C */
    for (i = 0; i < _PB_NI; i++){
      bbop_op(BBOP_MUL, C[i], array_beta, C[i], _PB_NI, 0, MUL_SSL_RB_34);

      for (j = 0; j < _PB_NJ; j++){

        for(k = 0; k < _PB_NK; ++k){
          transposed_b[k] = B[k][j];
        }

        bbop_op(BBOP_MUL, A[i], transposed_b, output_a_b, _PB_NK, 1, MUL_SSL_RB_37);
        // bbop_op(BBOP_MUL, output_a_b, array_alpha, output_a_b_alpha, _PB_NK, 2, MUL_SSL_RB_37);
        // C[i][j] = bbop_op_red(BBOP_ADD, output_a_b_alpha, _PB_NK, 3);

      }
    }
  #endif 
}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int ni = NI;
  int nj = NJ;
  int nk = NK;

   #ifdef RUN_BASELINE
    omp_set_num_threads(THREADS);
  #endif 

  #ifdef RUN_PIM
    omp_set_num_threads(NUM_THREADS);
    initialize_bbop_statistics();
  #endif

  int n_threads = omp_get_max_threads();
  printf("Number of threads: %d\n", n_threads);

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,NI,NJ,ni,nj);
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NI,NK,ni,nk);
  POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,NK,NJ,nk,nj);

  if (INT4_DATA_TYPE == 1){
    printf("Running with INT4_DATA_TYPE\n");
  }

  /* Initialize array(s). */
  init_array (ni, nj, nk, &alpha, &beta,
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  #ifdef GET_TIME
    Timer timer;
    startTimer(&timer);
  #endif 

  #ifndef ENERGY_MINUS_KERNEL

  /* Run kernel. */
  kernel_gemm (ni, nj, nk,
	       alpha, beta,
	       POLYBENCH_ARRAY(C),
	       POLYBENCH_ARRAY(A),
	       POLYBENCH_ARRAY(B));

  #endif 

  #ifdef GET_TIME
    stopTimer(&timer);
    #ifdef RUN_BASELINE
      printElapsedTime(timer);
    #endif
  #endif 

  #ifdef RUN_PIM
    print_bbop_statistic();
  #endif
  
  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(ni, nj,  POLYBENCH_ARRAY(C)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}
