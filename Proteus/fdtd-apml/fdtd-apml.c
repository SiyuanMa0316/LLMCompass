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
/* Default data type is double, default size is 256x256x256. */
#include "fdtd-apml.h"


/* Array initialization. */
static
void init_array (int cz,
		 int cxm,
		 int cym,
		 DATA_TYPE *mui,
		 DATA_TYPE *ch,
		 DATA_TYPE POLYBENCH_2D(Ax,CZ+1,CYM+1,cz+1,cym+1),
		 DATA_TYPE POLYBENCH_2D(Ry,CZ+1,CYM+1,cz+1,cym+1),
		 DATA_TYPE POLYBENCH_3D(Ex,CZ+1,CYM+1,CXM+1,cz+1,cym+1,cxm+1),
		 DATA_TYPE POLYBENCH_3D(Ey,CZ+1,CYM+1,CXM+1,cz+1,cym+1,cxm+1),
		 DATA_TYPE POLYBENCH_3D(Hz,CZ+1,CYM+1,CXM+1,cz+1,cym+1,cxm+1),
		 DATA_TYPE POLYBENCH_1D(czm,CZ+1,cz+1),
		 DATA_TYPE POLYBENCH_1D(czp,CZ+1,cz+1),
		 DATA_TYPE POLYBENCH_1D(cxmh,CXM+1,cxm+1),
		 DATA_TYPE POLYBENCH_1D(cxph,CXM+1,cxm+1),
		 DATA_TYPE POLYBENCH_1D(cymh,CYM+1,cym+1),
		 DATA_TYPE POLYBENCH_1D(cyph,CYM+1,cym+1))
{
  int i, j, k;
  *mui = 2341;
  *ch = 42;
  for (i = 0; i <= cz; i++)
    {
      czm[i] = ((DATA_TYPE) i + 1) / cxm;
      czp[i] = ((DATA_TYPE) i + 2) / cxm;
    }
  for (i = 0; i <= cxm; i++)
    {
      cxmh[i] = ((DATA_TYPE) i + 3) / cxm;
      cxph[i] = ((DATA_TYPE) i + 4) / cxm;
    }
  for (i = 0; i <= cym; i++)
    {
      cymh[i] = ((DATA_TYPE) i + 5) / cxm;
      cyph[i] = ((DATA_TYPE) i + 6) / cxm;
    }

  for (i = 0; i <= cz; i++)
    for (j = 0; j <= cym; j++)
      {
	Ry[i][j] = ((DATA_TYPE) i*(j+1) + 10) / cym;
	Ax[i][j] = ((DATA_TYPE) i*(j+2) + 11) / cym;
	for (k = 0; k <= cxm; k++)
	  {
	    Ex[i][j][k] = ((DATA_TYPE) i*(j+3) + k + 1) / cxm;
	    Ey[i][j][k] = ((DATA_TYPE) i*(j+4) + k + 2) / cym;
	    Hz[i][j][k] = ((DATA_TYPE) i*(j+5) + k + 3) / cz;
	  }
      }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int cz,
		 int cxm,
		 int cym,
		 DATA_TYPE POLYBENCH_3D(Bza,CZ+1,CYM+1,CXM+1,cz+1,cym+1,cxm+1),
		 DATA_TYPE POLYBENCH_3D(Ex,CZ+1,CYM+1,CXM+1,cz+1,cym+1,cxm+1),
		 DATA_TYPE POLYBENCH_3D(Ey,CZ+1,CYM+1,CXM+1,cz+1,cym+1,cxm+1),
		 DATA_TYPE POLYBENCH_3D(Hz,CZ+1,CYM+1,CXM+1,cz+1,cym+1,cxm+1))
{
  int i, j, k;
  
  for (i = 0; i <= cz; i++)
    for (j = 0; j <= cym; j++)
      for (k = 0; k <= cxm; k++) {
	fprintf(stderr, DATA_PRINTF_MODIFIER, Bza[i][j][k]);
	fprintf(stderr, DATA_PRINTF_MODIFIER, Ex[i][j][k]);
	fprintf(stderr, DATA_PRINTF_MODIFIER, Ey[i][j][k]);
	fprintf(stderr, DATA_PRINTF_MODIFIER, Hz[i][j][k]);
	if ((i * cxm + j) % 20 == 0) fprintf(stderr, "\n");
      }
  fprintf(stderr, "\n");
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_fdtd_apml(int cz,
		      int cxm,
		      int cym,
		      DATA_TYPE mui,
		      DATA_TYPE ch,
		      DATA_TYPE POLYBENCH_2D(Ax,CZ+1,CYM+1,cz+1,cym+1),
		      DATA_TYPE POLYBENCH_2D(Ry,CZ+1,CYM+1,cz+1,cym+1),
		      DATA_TYPE POLYBENCH_2D(clf,CYM+1,CXM+1,cym+1,cxm+1),
		      DATA_TYPE POLYBENCH_2D(tmp,CYM+1,CXM+1,cym+1,cxm+1),
		      DATA_TYPE POLYBENCH_3D(Bza,CZ+1,CYM+1,CXM+1,cz+1,cym+1,cxm+1),
		      DATA_TYPE POLYBENCH_3D(Ex,CZ+1,CYM+1,CXM+1,cz+1,cym+1,cxm+1),
		      DATA_TYPE POLYBENCH_3D(Ey,CZ+1,CYM+1,CXM+1,cz+1,cym+1,cxm+1),
		      DATA_TYPE POLYBENCH_3D(Hz,CZ+1,CYM+1,CXM+1,cz+1,cym+1,cxm+1),
		      DATA_TYPE POLYBENCH_1D(czm,CZ+1,cz+1),
		      DATA_TYPE POLYBENCH_1D(czp,CZ+1,cz+1),
		      DATA_TYPE POLYBENCH_1D(cxmh,CXM+1,cxm+1),
		      DATA_TYPE POLYBENCH_1D(cxph,CXM+1,cxm+1),
		      DATA_TYPE POLYBENCH_1D(cymh,CYM+1,cym+1),
		      DATA_TYPE POLYBENCH_1D(cyph,CYM+1,cym+1))
{
  int iz, iy, ix;

#ifdef RUN_BASELINE
  printf("Running baseline.\n");

  #pragma scop
  #pragma omp parallel
    {
      #pragma omp for private (iy, ix)
      for (iz = 0; iz < _PB_CZ; iz++)
        {
	  for (iy = 0; iy < _PB_CYM; iy++)
	    {
	      for (ix = 0; ix < _PB_CXM; ix++)
		{
		  clf[iz][iy] = Ex[iz][iy][ix] - Ex[iz][iy+1][ix] + Ey[iz][iy][ix+1] - Ey[iz][iy][ix];

		  tmp[iz][iy] = (cymh[iy] / cyph[iy]) * Bza[iz][iy][ix] - (ch / cyph[iy]) * clf[iz][iy];
		 
		  if (cxph[ix] > 0){	
		  	Hz[iz][iy][ix] = (cxmh[ix] /cxph[ix]) * Hz[iz][iy][ix]
		    			+ (mui * czp[iz] / cxph[ix]) * tmp[iz][iy]
		    			- (mui * czm[iz] / cxph[ix]) * Bza[iz][iy][ix];
		  }
		  else{
		        Hz[iz][iy][ix] = 0;
		  }
		  Bza[iz][iy][ix] = tmp[iz][iy];
		}
	      clf[iz][iy] = Ex[iz][iy][_PB_CXM] - Ex[iz][iy+1][_PB_CXM] + Ry[iz][iy] - Ey[iz][iy][_PB_CXM];
	      tmp[iz][iy] = (cymh[iy] / cyph[iy]) * Bza[iz][iy][_PB_CXM] - (ch / cyph[iy]) * clf[iz][iy];
	      Hz[iz][iy][_PB_CXM]=(cxmh[_PB_CXM] / cxph[_PB_CXM]) * Hz[iz][iy][_PB_CXM]
		+ (mui * czp[iz] / cxph[_PB_CXM]) * tmp[iz][iy]
		- (mui * czm[iz] / cxph[_PB_CXM]) * Bza[iz][iy][_PB_CXM];
		
	      Bza[iz][iy][_PB_CXM] = tmp[iz][iy];

	      for (ix = 0; ix < _PB_CXM; ix++)
		{
		  clf[iz][iy] = Ex[iz][_PB_CYM][ix] - Ax[iz][ix] + Ey[iz][_PB_CYM][ix+1] - Ey[iz][_PB_CYM][ix];
		  
		  tmp[iz][iy] = (cymh[_PB_CYM] / cyph[iy]) * Bza[iz][iy][ix] - (ch / cyph[iy]) * clf[iz][iy];
		  
		 if (cxph[ix] > 0){
		  	Hz[iz][_PB_CYM][ix] = (cxmh[ix] / cxph[ix]) * Hz[iz][_PB_CYM][ix]
		    			      + (mui * czp[iz] / cxph[ix]) * tmp[iz][iy]
		    			      - (mui * czm[iz] / cxph[ix]) * Bza[iz][_PB_CYM][ix];
		 }
		 else{
			Hz[iz][iy][ix] = 0;
		}
		  Bza[iz][_PB_CYM][ix] = tmp[iz][iy];
		}
	      clf[iz][iy] = Ex[iz][_PB_CYM][_PB_CXM] - Ax[iz][_PB_CXM] + Ry[iz][_PB_CYM] - Ey[iz][_PB_CYM][_PB_CXM];
	      tmp[iz][iy] = (cymh[_PB_CYM] / cyph[_PB_CYM]) * Bza[iz][_PB_CYM][_PB_CXM] - (ch / cyph[_PB_CYM]) * clf[iz][iy];
	      Hz[iz][_PB_CYM][_PB_CXM] = (cxmh[_PB_CXM] / cxph[_PB_CXM]) * Hz[iz][_PB_CYM][_PB_CXM]
		+ (mui * czp[iz] / cxph[_PB_CXM]) * tmp[iz][iy]
		- (mui * czm[iz] / cxph[_PB_CXM]) * Bza[iz][_PB_CYM][_PB_CXM];
	      Bza[iz][_PB_CYM][_PB_CXM] = tmp[iz][iy];
	    }
	}
  }
  #pragma endscop
#endif 

#ifdef RUN_PIM

	DATA_TYPE tmp1[_PB_CXM];
	DATA_TYPE tmp2[_PB_CXM];
	DATA_TYPE tmp3[_PB_CXM];
	DATA_TYPE tmp4[_PB_CXM];
	DATA_TYPE tmp5[_PB_CXM];
	DATA_TYPE tmp6[_PB_CXM];
	DATA_TYPE tmp7[_PB_CXM];
	DATA_TYPE tmp8[_PB_CXM];
	DATA_TYPE tmp9[_PB_CXM];

	DATA_TYPE tmp14[_PB_CXM];
	DATA_TYPE tmp15[_PB_CXM];
	DATA_TYPE tmp16[_PB_CXM];
	DATA_TYPE tmp17[_PB_CXM];
	DATA_TYPE tmp18[_PB_CXM];
	DATA_TYPE tmp19[_PB_CXM];
	DATA_TYPE tmp20[_PB_CXM];
	DATA_TYPE tmp21[_PB_CXM];
	DATA_TYPE tmp22[_PB_CXM];

	DATA_TYPE aux_tmp[_PB_CXM];


	for (iz = 0; iz < _PB_CZ; iz++){
		for (iy = 0; iy < _PB_CYM; iy++){
			for (ix = 0; ix < _PB_CXM; ix++){
				tmp1[ix] = mui * czp[iz];
				tmp2[ix] = mui * czm[iz];
				aux_tmp[ix] = tmp[iz][iy];
			}

			// tmp3
			bbop_op(BBOP_DIV, tmp1, cxph, tmp3, _PB_CXM, 0, BBOP_DIV);
			//tmp4
			bbop_op(BBOP_DIV, tmp2, cxph, tmp4, _PB_CXM, 1, BBOP_DIV);
			//tmp5 
			bbop_op(BBOP_DIV, cxmh, cxph, tmp5, _PB_CXM, 2, BBOP_DIV);
			// tmp6
			bbop_op(BBOP_MUL, tmp5, Hz[iz][iy], tmp6, _PB_CXM, 3, BBOP_MUL);
			// tmp7
			bbop_op(BBOP_MUL, tmp3, aux_tmp, tmp7, _PB_CXM, 4, BBOP_MUL);
			// tmp8
			bbop_op(BBOP_MUL, tmp4, Bza[iz][iy], tmp8, _PB_CXM, 5, BBOP_MUL);
			//tmp9
			bbop_op(BBOP_ADD, tmp8, tmp7, tmp9, _PB_CXM, 6, BBOP_ADD);
			//out 
			bbop_op(BBOP_SUB, tmp8, tmp9, Hz[iz][iy], _PB_CXM, 7, BBOP_SUB);


			for (ix = 0; ix < _PB_CXM; ix++){
				tmp15[ix] = (mui * czp[iz]);  
				tmp16[ix] = (mui * czm[iz]); 
			}

			bbop_op(BBOP_DIV, cxmh, cxph, tmp14, _PB_CXM, 8, BBOP_DIV);
			bbop_op(BBOP_DIV, tmp15, cxph, tmp17, _PB_CXM, 9, BBOP_DIV);
			bbop_op(BBOP_DIV, tmp16, cxph, tmp18, _PB_CXM, 10, BBOP_DIV);
			bbop_op(BBOP_MUL, tmp14, Hz[iz][_PB_CYM], tmp19, _PB_CXM, 11, BBOP_MUL);
			bbop_op(BBOP_MUL, tmp17, aux_tmp, tmp20, _PB_CXM, 12,BBOP_MUL);
			bbop_op(BBOP_MUL, tmp18, Bza[iz][_PB_CYM], tmp21, _PB_CXM, 13, BBOP_MUL);
			bbop_op(BBOP_ADD, tmp19, tmp20, tmp22, _PB_CXM, 14, BBOP_ADD);
			bbop_op(BBOP_SUB, tmp22, tmp21, Hz[iz][_PB_CYM], _PB_CXM, 15, BBOP_SUB);
	    }
	}
#endif 
}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int cz = CZ;
  int cym = CYM;
  int cxm = CXM;

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
  DATA_TYPE mui;
  DATA_TYPE ch;
  POLYBENCH_2D_ARRAY_DECL(Ax,DATA_TYPE,CZ+1,CYM+1,cz+1,cym+1);
  POLYBENCH_2D_ARRAY_DECL(Ry,DATA_TYPE,CZ+1,CYM+1,cz+1,cym+1);
  POLYBENCH_2D_ARRAY_DECL(clf,DATA_TYPE,CYM+1,CXM+1,cym+1,cxm+1);
  POLYBENCH_2D_ARRAY_DECL(tmp,DATA_TYPE,CYM+1,CXM+1,cym+1,cxm+1);
  POLYBENCH_3D_ARRAY_DECL(Bza,DATA_TYPE,CZ+1,CYM+1,CXM+1,cz+1,cym+1,cxm+1);
  POLYBENCH_3D_ARRAY_DECL(Ex,DATA_TYPE,CZ+1,CYM+1,CXM+1,cz+1,cym+1,cxm+1);
  POLYBENCH_3D_ARRAY_DECL(Ey,DATA_TYPE,CZ+1,CYM+1,CXM+1,cz+1,cym+1,cxm+1);
  POLYBENCH_3D_ARRAY_DECL(Hz,DATA_TYPE,CZ+1,CYM+1,CXM+1,cz+1,cym+1,cxm+1);
  POLYBENCH_1D_ARRAY_DECL(czm,DATA_TYPE,CZ+1,cz+1);
  POLYBENCH_1D_ARRAY_DECL(czp,DATA_TYPE,CZ+1,cz+1);
  POLYBENCH_1D_ARRAY_DECL(cxmh,DATA_TYPE,CXM+1,cxm+1);
  POLYBENCH_1D_ARRAY_DECL(cxph,DATA_TYPE,CXM+1,cxm+1);
  POLYBENCH_1D_ARRAY_DECL(cymh,DATA_TYPE,CYM+1,cym+1);
  POLYBENCH_1D_ARRAY_DECL(cyph,DATA_TYPE,CYM+1,cym+1);

  exit(1);
  /* Initialize array(s). */
  init_array (cz, cxm, cym, &mui, &ch,
  	      POLYBENCH_ARRAY(Ax),
  	      POLYBENCH_ARRAY(Ry),
  	      POLYBENCH_ARRAY(Ex),
  	      POLYBENCH_ARRAY(Ey),
  	      POLYBENCH_ARRAY(Hz),
  	      POLYBENCH_ARRAY(czm),
  	      POLYBENCH_ARRAY(czp),
  	      POLYBENCH_ARRAY(cxmh),
  	      POLYBENCH_ARRAY(cxph),
  	      POLYBENCH_ARRAY(cymh),
  	      POLYBENCH_ARRAY(cyph));

  /* Start timer. */
  polybench_start_instruments;

  #ifdef GET_TIME
    Timer timer;
    startTimer(&timer);
  #endif 

  #ifndef ENERGY_MINUS_KERNEL
	
  /* Run kernel. */
  kernel_fdtd_apml (cz, cxm, cym, mui, ch,
  		    POLYBENCH_ARRAY(Ax),
  		    POLYBENCH_ARRAY(Ry),
  		    POLYBENCH_ARRAY(clf),
  		    POLYBENCH_ARRAY(tmp),
  		    POLYBENCH_ARRAY(Bza),
  		    POLYBENCH_ARRAY(Ex),
  		    POLYBENCH_ARRAY(Ey),
  		    POLYBENCH_ARRAY(Hz),
  		    POLYBENCH_ARRAY(czm),
  		    POLYBENCH_ARRAY(czp),
  		    POLYBENCH_ARRAY(cxmh),
  		    POLYBENCH_ARRAY(cxph),
  		    POLYBENCH_ARRAY(cymh),
  		    POLYBENCH_ARRAY(cyph));

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
  polybench_prevent_dce(print_array(cz, cxm, cym,
  				    POLYBENCH_ARRAY(Bza),
  				    POLYBENCH_ARRAY(Ex),
  				    POLYBENCH_ARRAY(Ey),
  				    POLYBENCH_ARRAY(Hz)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(Ax);
  POLYBENCH_FREE_ARRAY(Ry);
  POLYBENCH_FREE_ARRAY(clf);
  POLYBENCH_FREE_ARRAY(tmp);
  POLYBENCH_FREE_ARRAY(Bza);
  POLYBENCH_FREE_ARRAY(Ex);
  POLYBENCH_FREE_ARRAY(Ey);
  POLYBENCH_FREE_ARRAY(Hz);
  POLYBENCH_FREE_ARRAY(czm);
  POLYBENCH_FREE_ARRAY(czp);
  POLYBENCH_FREE_ARRAY(cxmh);
  POLYBENCH_FREE_ARRAY(cxph);
  POLYBENCH_FREE_ARRAY(cymh);
  POLYBENCH_FREE_ARRAY(cyph);

  return 0;
}

