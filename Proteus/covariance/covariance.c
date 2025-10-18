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


/* Include benchmark-specific header. */
/* Default data type is double, default size is 4000. */
#include "covariance.h"


/* Array initialization. */
static
void init_array (int m, int n,
		 DATA_TYPE *float_n,
		 DATA_TYPE POLYBENCH_2D(data,M,N,m,n))
{
  int i, j;

  *float_n = 1;

  for (i = 0; i < M; i++){
    for (j = 0; j < N; j++){
      data[i][j] = ((DATA_TYPE) i*j) / M;
    }
  }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m,
		 DATA_TYPE POLYBENCH_2D(symmat,M,M,m,m))

{
  int i, j;

  for (i = 0; i < m; i++)
    for (j = 0; j < m; j++) {
      fprintf (stderr, DATA_PRINTF_MODIFIER, symmat[i][j]);
      if ((i * m + j) % 20 == 0) fprintf (stderr, "\n");
    }
  fprintf (stderr, "\n");
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_covariance(int m, int n,
		       DATA_TYPE float_n,
		       DATA_TYPE POLYBENCH_2D(data,M,N,m,n),
		       DATA_TYPE POLYBENCH_2D(symmat,M,M,m,m),
		       DATA_TYPE POLYBENCH_1D(mean,M,m))
{
  int i, j, j1, j2;
#ifdef RUN_BASELINE
  printf("Running baseline\n");

  #pragma scop
  /* Determine mean of column vectors of input data matrix */
  #pragma omp parallel
  {
    #pragma omp for private (i)
    for (j = 0; j < _PB_M; j++){
      mean[j] = 0.0;
	    
      for (i = 0; i < _PB_N; i++) mean[j] += data[i][j];
	    
      mean[j] /= float_n;
    }
      
    /* Center the column vectors. */
    #pragma omp for private (j)
    for (i = 0; i < _PB_N; i++)
      for (j = 0; j < _PB_M; j++) data[i][j] -= mean[j];
      
    /* Calculate the m * m covariance matrix. */
    #pragma omp for private (j2, i)
    for (j1 = 0; j1 < _PB_M; j1++)
      for (j2 = j1; j2 < _PB_M; j2++){
        symmat[j1][j2] = 0.0;
	  
        for (i = 0; i < _PB_N; i++) symmat[j1][j2] += data[i][j1] * data[i][j2];
	      
        symmat[j2][j1] = symmat[j1][j2];
      }
  }
  #pragma endscop
#endif


#ifdef RUN_PIM
  printf("Running PIM\n");

  // print the data matrix
  
  unsigned int largest = 0;  
  for (j = 0; j < _PB_M; j++){
      mean[j] = 0;
	    
      for (i = 0; i < _PB_N; i++){ 
	      mean[j] += data[i][j];
      }
	    
  }

  /* Determine mean of column vectors of input data matrix */
  for (j = 0; j < _PB_M; j++){
    mean[j] = bbop_op_red(BBOP_ADD, data[i], _PB_N, 0);
  }

  // make float_n an array so that it can be used in the reduction
  DATA_TYPE float_n_array[_PB_M];
  for (j = 0; j < _PB_M; j++){
    float_n_array[j] = float_n;
  }

  //print the mean vector

  for(i = 0; i < _PB_M; i++) largest = (mean[j] > largest)? mean[j] : largest;

  bbop_op(BBOP_DIV, mean, float_n_array, mean, _PB_M, 1, MUL_SSL_DIV_8);
      
   /* Center the column vectors. */
  for (i = 0; i < _PB_N; i++){
    bbop_op(BBOP_SUB, data[i], mean, data[i], _PB_M, 2, SUB_SSL_FA_11);
  }

#endif 

}

int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int m = M;

  #ifdef RUN_BASELINE
    omp_set_num_threads(THREADS);
  #endif 

  #ifdef RUN_PIM
    omp_set_num_threads(1);
    initialize_bbop_statistics();
  #endif

  int n_threads = omp_get_max_threads();
  printf("Number of threads: %d\n", n_threads);

  /* Variable declaration/allocation. */
  DATA_TYPE float_n;
  POLYBENCH_2D_ARRAY_DECL(data,DATA_TYPE,M,N,m,n);
  POLYBENCH_2D_ARRAY_DECL(symmat,DATA_TYPE,M,M,m,m);
  POLYBENCH_1D_ARRAY_DECL(mean,DATA_TYPE,M,m);
  
  /* Initialize array(s). */
  init_array (m, n, &float_n, POLYBENCH_ARRAY(data));
  
  /* Start timer. */
  polybench_start_instruments;

  #ifdef GET_TIME
    Timer timer;
    startTimer(&timer);
  #endif 

  #ifndef ENERGY_MINUS_KERNEL

    /* Run kernel. */
    kernel_covariance (m, n, float_n,
          POLYBENCH_ARRAY(data),
          POLYBENCH_ARRAY(symmat),
          POLYBENCH_ARRAY(mean));

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
  polybench_prevent_dce(print_array(m, POLYBENCH_ARRAY(symmat)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(data);
  POLYBENCH_FREE_ARRAY(symmat);
  POLYBENCH_FREE_ARRAY(mean);

  return 0;
}
