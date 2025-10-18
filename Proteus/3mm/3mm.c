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
#include "3mm.h"


/* Array initialization. */
static
void init_array(int ni, int nj, int nk, int nl, int nm,
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
		DATA_TYPE POLYBENCH_2D(C,NJ,NM,nj,nm),
		DATA_TYPE POLYBENCH_2D(D,NM,NL,nm,nl))
{
  int i, j;

  for (i = 0; i < ni; i++){
    for (j = 0; j < nk; j++){
      DATA_TYPE tmp = ((DATA_TYPE) i*j / ni);
      // printf("%d * %d / %d = %d\n", i, j, ni, tmp);

      if (INT4_DATA_TYPE == 1){
        tmp = tmp % 16;
      }

      A[i][j] = tmp;
      // printf("A[%d][%d] = %0.2lf\n", i, j, A[i][j]);
    }
  }

  for (i = 0; i < nk; i++){
    for (j = 0; j < nj; j++){
      DATA_TYPE tmp = ((DATA_TYPE) i*(j+1)) / nj;

      if (INT4_DATA_TYPE == 1){
        tmp = tmp % 16;
      }

      B[i][j] = tmp;
    }
  }

  printf("A matrix:\n");
    for (i = 0; i < _PB_NI; i++){
        for (int k = 0; k < _PB_NK; k++){
            printf("%d\t", A[i][k]);
        }
        printf("\n");
    }
    // print B matrix
    printf("B matrix:\n");
    for (int k = 0; k < _PB_NK; k++){
        for (j = 0; j < _PB_NJ; j++){
            printf(" %d\t", B[k][j]);
        }
        printf("\n");
    } 

  for (i = 0; i < nj; i++){
    for (j = 0; j < nm; j++){
      DATA_TYPE tmp = ((DATA_TYPE) i*(j+3)) / nl;

      if (INT4_DATA_TYPE == 1){
        tmp = tmp % 16;
      }

      C[i][j] = tmp;
    }
  }

  for (i = 0; i < nm; i++){
    for (j = 0; j < nl; j++){
      DATA_TYPE tmp = ((DATA_TYPE) i*(j+2)) / nk;

      if (INT4_DATA_TYPE == 1){
        tmp = tmp % 16;
      }

      D[i][j] = tmp;
    }
  }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, int nl,
		 DATA_TYPE POLYBENCH_2D(G,NI,NL,ni,nl))
{
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++) {
	fprintf (stderr, DATA_PRINTF_MODIFIER, G[i][j]);
	if ((i * ni + j) % 20 == 0) fprintf (stderr, "\n");
    }
  fprintf (stderr, "\n");
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_3mm(int ni, int nj, int nk, int nl, int nm,
		DATA_TYPE POLYBENCH_2D(E,NI,NJ,ni,nj),
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
		DATA_TYPE POLYBENCH_2D(F,NJ,NL,nj,nl),
		DATA_TYPE POLYBENCH_2D(C,NJ,NM,nj,nm),
		DATA_TYPE POLYBENCH_2D(D,NM,NL,nm,nl),
		DATA_TYPE POLYBENCH_2D(G,NI,NL,ni,nl))
{
  int i, j, k;
 
  # ifdef RUN_BASELINE
    printf("Running baseline version\n");

    #pragma scop
    #pragma omp parallel private (j, k)
    {
      /* E := A*B */
      #pragma omp for
      for (i = 0; i < _PB_NI; i++)
        for (j = 0; j < _PB_NJ; j++){
            E[i][j] = 0;
      
          for (k = 0; k < _PB_NK; ++k)
            E[i][j] += A[i][k] * B[k][j];
        }
      
      /* F := C*D */
      #pragma omp for
      for (i = 0; i < _PB_NJ; i++)
        for (j = 0; j < _PB_NL; j++){
          F[i][j] = 0;
      
          for (k = 0; k < _PB_NM; ++k)
            F[i][j] += C[i][k] * D[k][j];
        }
      
      /* G := E*F */
      #pragma omp for
      for (i = 0; i < _PB_NI; i++)
        for (j = 0; j < _PB_NL; j++){
          G[i][j] = 0;
      
          for (k = 0; k < _PB_NJ; ++k)
            G[i][j] += E[i][k] * F[k][j];
        }
    }
    #pragma endscop
  # endif 

  #ifdef RUN_PIM
    printf("Running PIM version\n");

    DATA_TYPE output_a_b[_PB_NK];
    DATA_TYPE output_c_d[_PB_NM];
    DATA_TYPE output_e_f[_PB_NJ];

    // B needs to be transposed
    DATA_TYPE b_transpose[_PB_NK];
    // C needs to be transposed
    DATA_TYPE d_transpose[_PB_NM];
    // E needs to be transposed
    DATA_TYPE f_transpose[_PB_NJ];

    /* E := A*B */
    for (i = 0; i < _PB_NI; i++){
        for (j = 0; j < _PB_NJ; j++){
            for (k = 0; k < _PB_NK; ++k){
                b_transpose[k] = B[k][j];
            }

            bbop_op(BBOP_MUL, A[i], b_transpose, output_a_b, _PB_NK, 0, MUL_SSL_Sklansky_22);
            // print_array(1, _PB_NK, output_a_b);
            E[i][j] = bbop_op_red(BBOP_ADD, output_a_b, _PB_NK, 1);
            if(INT4_DATA_TYPE == 1)
            {
                E[i][j] = E[i][j] % 16;
            }

        }
    }
    // print A matrix
    printf("A matrix:\n");
    for (i = 0; i < _PB_NI; i++){
        for (k = 0; k < _PB_NK; k++){
            printf("%d\t", A[i][k]);
        }
        printf("\n");
    }
    // print B matrix
    printf("B matrix:\n");
    for (k = 0; k < _PB_NK; k++){
        for (j = 0; j < _PB_NJ; j++){
            printf(" %d\t", B[k][j]);
        }
        printf("\n");
    } 
    
    // print E matrix
    printf("E matrix:\n");
    for (i = 0; i < _PB_NI; i++){
        for (j = 0; j < _PB_NJ; j++){
            printf(" %d\t", E[i][j]);
        }
        printf("\n");
    }
    
    /* F := C*D */
    for (i = 0; i < _PB_NJ; i++){
        for (j = 0; j < _PB_NL; j++){
           
            for (k = 0; k < _PB_NM; ++k){
                d_transpose[k] = D[k][j];
            }

            bbop_op(BBOP_MUL, C[i], d_transpose, output_c_d, _PB_NM, 2, MUL_SSL_Sklansky_22);
            F[i][j] = bbop_op_red(BBOP_ADD, output_c_d, _PB_NM, 3);
            if (INT4_DATA_TYPE == 1){
                F[i][j] = F[i][j] % 16;
            }

        }
    }

    /* G := E*F */
    for (i = 0; i < _PB_NI; i++){
      for (j = 0; j < _PB_NL; j++){
            for (k = 0; k < _PB_NJ; ++k){
                f_transpose[k] = F[k][j];
            }

            bbop_op(BBOP_MUL, E[i], f_transpose, output_e_f, _PB_NJ, 4, MUL_SSL_Sklansky_15);
            G[i][j] = bbop_op_red(BBOP_ADD, output_e_f, _PB_NJ, 5);
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
  int nl = NL;
  int nm = NM;

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
  POLYBENCH_2D_ARRAY_DECL(E, DATA_TYPE, NI, NJ, ni, nj);
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NI, NK, ni, nk);
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, NK, NJ, nk, nj);
  POLYBENCH_2D_ARRAY_DECL(F, DATA_TYPE, NJ, NL, nj, nl);
  POLYBENCH_2D_ARRAY_DECL(C, DATA_TYPE, NJ, NM, nj, nm);
  POLYBENCH_2D_ARRAY_DECL(D, DATA_TYPE, NM, NL, nm, nl);
  POLYBENCH_2D_ARRAY_DECL(G, DATA_TYPE, NI, NL, ni, nl);

  if (INT4_DATA_TYPE == 1){
    printf("Running with INT4_DATA_TYPE\n");
  }
  

  /* Initialize array(s). */
  init_array (ni, nj, nk, nl, nm,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B),
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(D));


  /* Start timer. */
  polybench_start_instruments;

  #ifdef GET_TIME
    Timer timer;
    startTimer(&timer);
  #endif 

  #ifndef ENERGY_MINUS_KERNEL

    /* Run kernel. */
    kernel_3mm (ni, nj, nk, nl, nm,
          POLYBENCH_ARRAY(E),
          POLYBENCH_ARRAY(A),
          POLYBENCH_ARRAY(B),
          POLYBENCH_ARRAY(F),
          POLYBENCH_ARRAY(C),
          POLYBENCH_ARRAY(D),
          POLYBENCH_ARRAY(G));
  #endif 
  
  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;
  

  #ifdef GET_TIME
    stopTimer(&timer);
    #ifdef RUN_BASELINE
      printElapsedTime(timer);
    #endif
  #endif 

  #ifdef RUN_PIM
    print_bbop_statistic();
  #endif


  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(ni, nl,  POLYBENCH_ARRAY(G)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(E);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);
  POLYBENCH_FREE_ARRAY(F);
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(D);
  POLYBENCH_FREE_ARRAY(G);

  return 0;
}
