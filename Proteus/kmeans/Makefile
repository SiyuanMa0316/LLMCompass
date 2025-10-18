# --- Compiler (override from CLI: make CC=gcc-15) ---
CC ?= cc

# --- OS detection (for getopt.c) ---
UNAME_S := $(shell uname -s)

# --- Common flags you can extend/override from CLI ---
CPPFLAGS ?=
CFLAGS   ?= -O3 -march=native -g0 -w
LDFLAGS  ?=
LDLIBS   ?=

# --- OpenMP flags: GCC vs (Apple) Clang ---
ifneq (,$(findstring clang,$(shell $(CC) --version)))
  # Apple/LLVM Clang needs these for OpenMP (requires libomp installed)
  OMP_CFLAGS := -Xpreprocessor -fopenmp
  OMP_LDLIBS := -lomp
else
  OMP_CFLAGS := -fopenmp
  OMP_LDLIBS :=
endif

CFLAGS += $(OMP_CFLAGS)
LDLIBS += $(OMP_LDLIBS)

# --- Build local getopt.c only on non-macOS ---
ifeq ($(UNAME_S),Darwin)
  GETOPT_SRC :=
else
  GETOPT_SRC := getopt.c
endif

# --- Sources ---
SOURCES_COMMON := cluster.c kmeans.c kmeans_clustering.c $(GETOPT_SRC)

# --- Target-specific defines (kept from your original) ---
BASELINE_DEFS := -DGET_TIME_FUNCTION -DRUN_BASELINE
PIM_DEFS      := -DRUN_PIM

# --- Targets ---
.PHONY: all clean
all: kmeans_baseline.exe kmeans_pim.exe

kmeans_baseline.exe: $(SOURCES_COMMON)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) $(BASELINE_DEFS) \
	    $(SOURCES_COMMON) -o $@ $(LDLIBS)

kmeans_pim.exe: $(SOURCES_COMMON) ../util/bbop_manager.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) $(PIM_DEFS) \
	    $(SOURCES_COMMON) ../util/bbop_manager.c -o $@ -lm $(LDLIBS)

clean:
	rm -f *.o *~ kmeans kmeans_baseline.exe kmeans_pim.exe
