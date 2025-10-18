/**
 * 3mm.h: This file is part of the PolyBench/C 3.2 test suite.
 *
 *
 * Contact: Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://polybench.sourceforge.net
 */
#ifndef _3MM_H
# define _3MM_H

/* Default to STANDARD_DATASET. */
# if !defined(MINI_DATASET) && !defined(SMALL_DATASET) && !defined(LARGE_DATASET) && !defined(EXTRALARGE_DATASET) && !defined(PAPER_DATASET) && !defined(PAPER_DATASET_V2) && !defined(PAPER_DATASET_V3)
#  define STANDARD_DATASET
# endif

/* Do not define anything if the user manually defines the size. */
# if !defined(NI) && !defined(NJ) && !defined(NK)
/* Define the possible dataset sizes. */
#  ifdef MINI_DATASET
#   define NI 32
#   define NJ 32
#   define NK 32
#   define NL 32
#   define NM 32
#  endif

#  ifdef SMALL_DATASET
#   define NI 128
#   define NJ 128
#   define NK 128
#   define NL 128
#   define NM 128
#  endif

#  ifdef STANDARD_DATASET /* Default if unspecified. */
#   define NI 1024
#   define NJ 1024
#   define NK 1024
#   define NL 1024
#   define NM 1024
#  endif

#  ifdef LARGE_DATASET
#   define NI 2000
#   define NJ 2000
#   define NK 2000
#   define NL 2000
#   define NM 2000
#  endif

#  ifdef EXTRALARGE_DATASET
#   define NI 4000
#   define NJ 4000
#   define NK 4000
#   define NL 4000
#   define NM 4000
#  endif

#  ifdef PAPER_DATASET
#   define NI 32000
#   define NJ 32000
#   define NK 32000
#   define NL 32000
#   define NM 32000
#  endif

#  ifdef PAPER_DATASET_V2
#   define NI 16000
#   define NJ 16000
#   define NK 16000
#   define NL 16000
#   define NM 16000
#  endif

#  ifdef PAPER_DATASET_V3
#   define NI 16000
#   define NJ 16000
#   define NK 32000
#   define NL 16000
#   define NM 32000
#  endif
# endif /* !N */

# define _PB_NI POLYBENCH_LOOP_BOUND(NI,ni)
# define _PB_NJ POLYBENCH_LOOP_BOUND(NJ,nj)
# define _PB_NK POLYBENCH_LOOP_BOUND(NK,nk)
# define _PB_NL POLYBENCH_LOOP_BOUND(NL,nl)
# define _PB_NM POLYBENCH_LOOP_BOUND(NM,nm)

# ifndef DATA_TYPE
#  define DATA_TYPE unsigned int
#  define DATA_PRINTF_MODIFIER "%0.2lf "
# endif

# ifndef INT4_DATA_TYPE
#  define INT4_DATA_TYPE 0
#endif 
#endif /* !_3MM */
