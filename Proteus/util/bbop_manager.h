#ifndef BBOP_H
#define BBOP_H


#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <omp.h>
#include <stdint.h>


#ifndef NUM_THREADS
    #define NUM_THREADS 32
#endif

#ifndef OUT_APPEND
    #define OUT_APPEND 0
#endif

#ifndef INT4_DATA_TYPE
    #define INT4_DATA_TYPE 0
#endif

#ifndef INT4_MAX
    #define INT4_MAX 15

#endif

#ifndef INT4_MIN
    #define INT4_MIN 0
#endif

#define FLOAT_TO_INT(x) ((x)>=0?(int)((x)+0.5):(int)((x)-0.5))

#define DATATYPE_BBOP unsigned int

// define the maxmium number bbops that can be executed
#define MAX_BBOPS 100

#define SIMD_WIDTH 65536.0
#define SUBARRAYS 64

#define AAP_ENERGY 0.871
#define AAP_LATENCY 49 // 49 ns

#define mmax(x, y) (x > y ? x : y)

// Create an enum to represent different operations
enum bbop_operation{
    BBOP_ADD,
    BBOP_ADD_8,
    BBOP_ADD_16,
    BBOP_ADD_32,
    BBOP_ADD_64,
    BBOP_SUB,
    BBOP_SUB_8,
    BBOP_SUB_16,
    BBOP_SUB_32,
    BBOP_SUB_64,
    BBOP_MUL,
    BBOP_MUL_8,
    BBOP_MUL_16,
    BBOP_MUL_32,
    BBOP_MUL_64,
    BBOP_DIV,
    BBOP_DIV_8,
    BBOP_DIV_16,
    BBOP_DIV_32,
    BBOP_DIV_64,
    BBOP_CPY,
    BBOP_RED,
    CPU_CPY,
    MUL_SSL_Sklansky_24,
    MUL_SSL_RB_39,
    MUL_SSL_RB_35,
    MUL_SSL_Sklansky_20,
    MUL_SSL_Sklansky_22,
    MUL_SSL_Sklansky_15,
    MUL_SSL_Sklansky_19,
    MUL_SSL_RB_34,
    MUL_SSL_RB_37,
    MUL_SSL_DIV_12,
    MUL_SSL_DIV_8,
    SUB_SSL_FA_11,
    MUL_SSL_Sklansky_18,
    SUB_SSL_FA_7,
    MUL_SSL_CS_14,
    SUB_SSL_FA_9,
    BBOP_RELU,
    BBOP_COUNT};
typedef enum bbop_operation bbop_operation;

typedef struct{
    int val;
    char pad[128];
} tvals;


struct bbop_statistic{
    bbop_operation operation;
    
    double simdram_1_subarray_latency;
    double simdram_64_subarray_latency;
    double simdram_64_subarray_dynamic_precision_latency;
    double daftpum_static_latency_optimized_latency;
    double daftpum_static_energy_optimized_latency;
    double daftpum_latency_optimized_latency;
    double daftpum_energy_optimized_latency;
    double daftpum_tfaw_enabled_latency;

    double simdram_1_subarray_energy;
    double simdram_64_subarray_energy;
    double simdram_64_subarray_dynamic_precision_energy;
    double daftpum_static_latency_optimized_energy;
    double daftpum_static_energy_optimized_energy;
    double daftpum_latency_optimized_energy;
    double daftpum_energy_optimized_energy;
    double daftpum_tfaw_enabled_energy;
    
    long largest_input_a;
    long largest_input_b;
};
typedef struct bbop_statistic bbop_statistic;



// Create an array to store the bbop_statistic
void initialize_bbop_statistics();

// Create a bbop_op function
void bbop_op(bbop_operation operation, DATATYPE_BBOP *A, DATATYPE_BBOP *B, DATATYPE_BBOP *C, unsigned long long size, int bbop_id, bbop_operation daftpum_operation);

DATATYPE_BBOP bbop_op_red(bbop_operation operation, DATATYPE_BBOP *A, long size, int bbop_id);

void print_bbop_statistic();

unsigned short int float2fix(float n);

int get_daftpum_adder_latency(int bit_precision, int size, char * adder_type, bool tfaw_enabled);
int get_daftpum_adder_energy(int bit_precision, int size, char * adder_type, bool tfaw_enabled);
int get_simdram_adder_latency(int bit_precision, int size, bool dynamic_bit_precision_enabled, bool salp_enabled);
int get_simdram_adder_energy(int bit_precision, int size, bool dynamic_bit_precision_enabled, bool salp_enabled);

int get_relu_latency(int bit_precision, int size, bool dynamic_bit_precision_enabled, bool salp_enabled);
int get_relu_energy(int bit_precision, int size, bool dynamic_bit_precision_enabled, bool salp_enabled);

int get_power_of_two_bit_precision(int bit_precision);

#endif
