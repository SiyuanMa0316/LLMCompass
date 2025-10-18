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
#include "doitgen.h"


/* Array initialization. */
static
void init_array(int nr, int nq, int np,
		DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np),
		DATA_TYPE POLYBENCH_2D(C4,NP,NP,np,np))
{
  long int i, j, k;
  printf("nr = %d, nq = %d, np = %d \n", nr, nq, np);

  for (i = 0; i < nr; i++){
    for (j = 0; j < nq; j++){
      for (k = 0; k < np; k++){
	//printf(" (i = %ld, j = %ld, k = %ld) \n", i, j, k); 
	A[i][j][k] = ((DATA_TYPE) i*j + k) / np;
      }
     }
   }

  for (i = 0; i < np; i++){
    for (j = 0; j < np; j++){
      C4[i][j] = ((DATA_TYPE) i*j) / np;
    }
  }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int nr, int nq, int np,
		 DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np))
{
  int i, j, k;

  for (i = 0; i < nr; i++)
    for (j = 0; j < nq; j++)
      for (k = 0; k < np; k++) {
	fprintf (stderr, DATA_PRINTF_MODIFIER, A[i][j][k]);
	if (i % 20 == 0) fprintf (stderr, "\n");
      }
  fprintf (stderr, "\n");
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_doitgen(int nr, int nq, int np,
		    DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np),
		    DATA_TYPE POLYBENCH_2D(C4,NP,NP,np,np),
		    DATA_TYPE POLYBENCH_3D(sum,NR,NQ,NP,nr,nq,np))
{
  int r, q, p, s;

  #ifdef RUN_BASELINE 
    printf("Running baseline\n"); 

    #pragma scop
    #pragma omp parallel
    {
      #pragma omp for private (q, p, s)
      for (r = 0; r < _PB_NR; r++){
        for (q = 0; q < _PB_NQ; q++){
          for (p = 0; p < _PB_NP; p++){
            sum[r][q][p] = 0;

            for (s = 0; s < _PB_NP; s++) sum[r][q][p] = sum[r][q][p] + A[r][q][s] * C4[s][p];
          }
      
          for (p = 0; p < _PB_NR; p++) A[r][q][p] = sum[r][q][p];
        }
      }
    }
    #pragma endscop
  #endif

  #ifdef RUN_PIM
    printf("Running PIM\n");

    DATA_TYPE output_a_c4[_PB_NP];
    DATA_TYPE transposed_c4[_PB_NP];

    for (r = 0; r < _PB_NR; r++){
      for (q = 0; q < _PB_NQ; q++){
        for (p = 0; p < _PB_NP; p++){
          // C4 needs to be transposed
          for (s = 0; s < _PB_NP; s++){
            transposed_c4[s] = C4[p][s];
          }

          bbop_op(BBOP_MUL_32, A[r][q], transposed_c4, output_a_c4, _PB_NP, 0, MUL_SSL_Sklansky_19);
          
	        for (s = 0; s < _PB_NP; s++) sum[r][q][p] += output_a_c4[s];
	        sum[r][q][p] = bbop_op_red(BBOP_ADD, output_a_c4, _PB_NP, 1);
        }
          
        bbop_op(BBOP_CPY, sum[r][q], sum[r][q], A[r][q], _PB_NP, 3, BBOP_CPY);
      }
    }

  #endif

}

int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int nr = NR;
  int nq = NQ;
  int np = NP;

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
  POLYBENCH_3D_ARRAY_DECL(A,DATA_TYPE,NR,NQ,NP,nr,nq,np);
  POLYBENCH_3D_ARRAY_DECL(sum,DATA_TYPE,NR,NQ,NP,nr,nq,np);
  POLYBENCH_2D_ARRAY_DECL(C4,DATA_TYPE,NP,NP,np,np);


  /* Initialize array(s). */
  init_array (nr, nq, np,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(C4));

        /* Start timer. */
  polybench_start_instruments;

  #ifdef GET_TIME
    Timer timer;
    startTimer(&timer);
  #endif 

  #ifndef ENERGY_MINUS_KERNEL

  /* Run kernel. */
  kernel_doitgen (nr, nq, np,
		  POLYBENCH_ARRAY(A),
		  POLYBENCH_ARRAY(C4),
		  POLYBENCH_ARRAY(sum));
  
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
  polybench_prevent_dce(print_array(nr, nq, np,  POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(sum);
  POLYBENCH_FREE_ARRAY(C4);

  return 0;
}
