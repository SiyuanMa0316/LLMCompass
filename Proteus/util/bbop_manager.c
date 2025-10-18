#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include "bbop_manager.h"
 
// Helper data structures
int latency_simdram_ripple_carry_adder[64] = {-1,    833,   833,   1225,  1617,  2009,  2401,  2793,  3185,  3577,  3969,  4361,  4753,  5145,  5537,  5929, 
                                      6321,  6713,  7105,  7497,  7889,  8281,  8673,  9065,  9457,  9849,  10241, 10633, 11025, 11417, 11809, 12201, 
                                      12593, 12985, 13377, 13769, 14161, 14553, 14945, 15337, 15729, 16121, 16513, 16905, 17297, 17689, 18081, 18473, 
                                      18865, 19257, 19649, 20041, 20433, 20825, 21217, 21609, 22001, 22393, 22785, 23177, 23569, 23961, 24353, 24745};

int latency_daftpum_full_adder[64] =  {-1,     676,  676,    915,   1154,  1393,  1632,  1871,  2110,  2349,  2588,  2827,  3066,  3305,  3544, 3783, 
                                4022,  4261,  4500,  4739,  4978,  5217,  5456,  5695,  5934,  6173,  6412,  6651,  6890,  7129,  7368, 7607, 
                                7846,  8085,  8324,  8563,  8802,  9041,  9280,  9519,  9758,  9997,  10236, 10475, 10714, 10953, 11192, 11431, 
                                11670, 11909, 12148, 12387, 12626, 12865, 13104, 13343, 13582, 13821, 14060, 14299, 14538, 14777, 15016, 15255};

int latency_daftpum_sklansky_adder[64] = {-1, 1757, 1757, 2556, 2556, 3152, 3152, 3152, 3152, 3812, 3812, 3812, 3812, 3812, 3812, 3812, 
                                3812, 4600, 4600, 4600, 4600, 4600, 4600, 4600, 4600, 4600, 4600, 4600, 4600, 4600, 4600, 4600,
                                4600, 5644, 5644, 5644, 5644, 5644, 5644, 5644, 5644, 5644, 5644, 5644, 5644, 5644, 5644, 5644, 
                                5644, 5644, 5644, 5644, 5644, 5644, 5644, 5644, 5644, 5644, 5644, 5644, 5644, 5644, 5644, 5644};

int latency_daftpum_koggee_adder[64] = {-1, 1663, 1663, 2227, 2227, 2823, 2823, 2823, 2823, 3483, 3483, 3483, 3483, 3483, 3483, 3483, 
                                3483, 4271, 4271, 4271, 4271, 4271, 4271, 4271, 4271, 4271, 4271, 4271, 4271, 4271, 4271, 4271, 
                                4271, 5315, 5315, 5315, 5315, 5315, 5315, 5315, 5315, 5315, 5315, 5315, 5315, 5315, 5315, 5315, 
                                5315, 5315, 5315, 5315, 5315, 5315, 5315, 5315, 5315, 5315, 5315, 5315, 5315, 5315, 5315, 5315};

int latency_daftpum_carryselect_adder[64] = {-1,   676,   676, 915,  1154, 2398, 2398, 2398, 2398, 3046, 3046, 3046, 3046, 3046, 3046, 3046, 
                                     3046, 4342, 4342, 4342, 4342, 4342, 4342, 4342, 4342, 4342, 4342, 4342, 4342, 4342, 4342, 4342, 
                                     4342, 6934, 6934, 6934, 6934, 6934, 6934, 6934, 6934, 6934, 6934, 6934, 6934, 6934, 6934, 6934, 
                                     6934, 6934, 6934, 6934, 6934, 6934, 6934, 6934, 6934, 6934, 6934, 6934, 6934, 6934, 6934, 6934};

int latency_daftpum_rbr_adder[64] = {-1,  2194, 2194, 2194, 2194, 2194, 2194, 2194, 2194, 2194, 2194, 2194, 2194, 2194, 2194, 2194, 
                            	    2194, 2194, 2194, 2194, 2194, 2194, 2194, 2194, 2194, 2194, 2194, 2194, 2194, 2194, 2194, 2194, 
                            	    2194, 2194, 2194, 2194, 2194, 2194, 2194, 2194, 2194, 2194, 2194, 2194, 2194, 2194, 2194, 2194, 
                            	    2194, 2194, 2194, 2194, 2194, 2194, 2194, 2194, 2194, 2194, 2194, 2194, 2194, 2194, 2194, 2194};        


int latency_simdram_ripple_carry_multiplier[64] = {
        -1, 1617, 1617, 4067, 7595, 12201, 17885, 24647, 32487, 41405,
        51401, 62475, 74627, 87857, 102165, 117551, 134015, 151557, 170177,
        189875, 210651, 232505, 255437, 279447, 304535, 330701, 357945, 386267,
        415667, 446145, 477701, 510335, 544047, 578837, 614705, 651651, 689675,
        728777, 768957, 810215, 852551, 895965, 940457, 986027, 1032675, 1080401,
        1129205, 1179087, 1230047, 1282085, 1335201, 1389395, 1444667, 1501017,
        1558445, 1616951, 1676535, 1737197, 1798937, 1861755, 1925651, 1990625,
        2056677, 2123807
};

int latency_daftpum_sklansky_multiplier[64] = {
        -1, 5120, 5120, 8722, 12656, 16856, 21283, 25911, 30720, 35698,
        40833, 46117, 51542, 57104, 62797, 68617, 74560, 80623, 86803,
        93098, 99505, 106023, 112649, 119382, 126221, 133163, 140208,
        147355, 154602, 161949, 169394, 176937, 184576, 192311, 200142,
        208067, 216086, 224199, 232404, 240702, 249091, 257571, 266142,
        274803, 283554, 292395, 301325, 310343, 319450, 328644, 337927,
        347296, 356753, 366296, 375926, 385643, 395445, 405333, 415306,
        425364, 435508, 445736, 456049, 466447    
};

int latency_daftpum_csa_multiplier[64] = {
        -1, 2974, 2974, 7773, 11052, 14675, 18642, 22953, 27608, 32607,
        37950, 43637, 49668, 56043, 62762, 69825, 77232, 84983, 93078,
        101517, 110300, 119427, 128898, 138713, 148872, 159375, 170222,
        181413, 192948, 204827, 217050, 229617, 242528, 255783, 269382,
        283325, 297612, 312243, 327218, 342537, 358200, 374207, 390558,
        407253, 424292, 441675, 459402, 477473, 495888, 514647, 533750,
        553197, 572988, 593123, 613602, 634425, 655592, 677103, 698958,
        721157, 743700, 766587, 789818, 813393
};

int latency_daftpum_rbr_multiplier[64] = {
        -1, 11232, 11232, 16872, 22528, 28200, 33888, 39592, 45312, 51048,
        56800, 62568, 68352, 74152, 79968, 85800, 91648, 97512, 103392,
        109288, 115200, 121128, 127072, 133032, 139008, 145000, 151008,
        157032, 163072, 169128, 175200, 181288, 187392, 193512, 199648,
        205800, 211968, 218152, 224352, 230568, 236800, 243048, 249312,
        255592, 261888, 268200, 274528, 280872, 287232, 293608, 300000,
        306408, 312832, 319272, 325728, 332200, 338688, 345192, 351712,
        358248, 364800, 371368, 377952, 384552
};


int latency_simdram_relu[64] = {-1,   196,  343,  490,  637,  784,  931,  1078, 
                                1225, 1372, 1519, 1666, 1813, 1960, 2107, 2254,
                                2401, 2548, 2695, 2842, 2989, 3136, 3283, 3430,
                                3577, 3724, 3871, 4018, 4165, 4312, 4459, 4606,
                                4753, 4900, 5047, 5194, 5341, 5488, 5635, 5782,
                                5929, 6076, 6223, 6370, 6517, 6664, 6811, 6958,
                                7105, 7252, 7399, 7546, 7693, 7840, 7987, 8134,
                                8281, 8428, 8575, 8722, 8869, 9016, 9163, 9310}; 

int get_daftpum_adder_latency(int bit_precision, int size, char * adder_type, bool tfaw_enabled){
    int full_adder_simdram_latency = latency_simdram_ripple_carry_adder[bit_precision];
    int full_adder_latency = latency_daftpum_full_adder[bit_precision];
    int sklansky_adder_latency = latency_daftpum_sklansky_adder[bit_precision];
    int koggee_adder_latency = latency_daftpum_koggee_adder[bit_precision];
    int carryselect_adder_latency = latency_daftpum_carryselect_adder[bit_precision];
    int rbr_adder_latency = latency_daftpum_rbr_adder[bit_precision];

    // get the lowest latency
    int min_latency = full_adder_latency;

    // check if the adder_type array is empty
    // if so, we need to figure out the best adder type
    // Otherwise, we already know the adder type

     if(strcmp(adder_type, "") == 0){
        bool found_smaller = false;
        // check the adder with the lowest energy

        if (sklansky_adder_latency < min_latency){
            min_latency = sklansky_adder_latency;
            strcpy(adder_type, "brent_kung_adder");
            found_smaller = true;
        }
        if (koggee_adder_latency < min_latency){
            min_latency = koggee_adder_latency;
            strcpy(adder_type, "koggee_adder");
            found_smaller = true;
        }
        if (carryselect_adder_latency < min_latency){
            min_latency = carryselect_adder_latency;
            strcpy(adder_type, "carryselect_adder");
            found_smaller = true;
        }
        if (rbr_adder_latency < min_latency){
            min_latency = rbr_adder_latency;
            strcpy(adder_type, "rbr_adder");
            found_smaller = true;
        }
        if (full_adder_simdram_latency < min_latency){
            min_latency = full_adder_simdram_latency;
            strcpy(adder_type, "full_adder_simdram");
            found_smaller = true;
        }

        if (found_smaller == false){
            strcpy(adder_type, "full_adder");
        }
    }
    else{
        if (strcmp(adder_type, "full_adder") == 0){
            min_latency = full_adder_latency;
        }
        else if(strcmp(adder_type, "koggee_adder") == 0){
            min_latency = koggee_adder_latency;
        }
        else if(strcmp(adder_type, "brent_kung_adder") == 0){
            min_latency = sklansky_adder_latency;
        }
        else if(strcmp(adder_type, "carryselect_adder") == 0){
            min_latency = carryselect_adder_latency;
        }
        else if(strcmp(adder_type, "rbr_adder") == 0){
            min_latency = rbr_adder_latency;
        }
        else if (strcmp(adder_type, "full_adder_simdram") == 0){
            min_latency = full_adder_simdram_latency;
        }
        else{
            printf("Error: adder_type not recognized\n");
        }
    }

    if (tfaw_enabled == false){
        // get the closest power of two to the bit_precision
        //int closest_power_of_two = (int) pow(2, ceil(log2(bit_precision)));
        int closest_power_of_two =  1;
        int parallelism = SUBARRAYS/closest_power_of_two;

        int parallel_elements = parallelism * SIMD_WIDTH;

        int repetition_factor = (parallel_elements > size)? 1 : ceil(size/parallel_elements);

        min_latency = min_latency * repetition_factor;
    }
    else{
        // if number of bit_precision is larger than 4, there is no parallelism
        if (bit_precision >= 4){
            int parallel_elements = 1 * SIMD_WIDTH;
            int repetition_factor = (parallel_elements > size)? 1 : ceil(size/parallel_elements);
            min_latency = min_latency * repetition_factor;
        }
        else{
           // int closest_power_of_two = (int) pow(2, ceil(log2(bit_precision)));
            int closest_power_of_two = 1;
            int parallelism = 4/closest_power_of_two;
            int parallel_elements = parallelism * SIMD_WIDTH;
            int repetition_factor = (parallel_elements > size)? 1 : ceil(size/parallel_elements);
            min_latency = min_latency * repetition_factor;
        }
    }

    return min_latency;
}

int get_simdram_adder_latency(int bit_precision, int size, bool dynamic_bit_precision_enabled, bool salp_enabled){
    if (dynamic_bit_precision_enabled){

        if (salp_enabled){
            int parallel_elements = SUBARRAYS * SIMD_WIDTH;
            int repetition_factor = (parallel_elements > size)? 1 : ceil(size/parallel_elements);
            
            return repetition_factor*latency_simdram_ripple_carry_adder[bit_precision];
        }
        else{
            int parallel_elements = 1 * SIMD_WIDTH;
            int repetition_factor = (parallel_elements > size)? 1 : ceil(size/parallel_elements);
            
            return repetition_factor*latency_simdram_ripple_carry_adder[bit_precision];
        }
    }
    else{
        if (salp_enabled){
            int parallel_elements = SUBARRAYS * SIMD_WIDTH;
            int repetition_factor = (parallel_elements > size)? 1 : ceil(size/parallel_elements);
            
            return repetition_factor*latency_simdram_ripple_carry_adder[32];
        }
        else{
            int parallel_elements = 1 * SIMD_WIDTH;
            int repetition_factor = (parallel_elements > size)? 1 : ceil(size/parallel_elements);
            
            return repetition_factor*latency_simdram_ripple_carry_adder[32];
        }
    }
}

int get_relu_latency(int bit_precision, int size, bool dynamic_bit_precision_enabled, bool salp_enabled){
    if (dynamic_bit_precision_enabled){
        if (salp_enabled){
            int parallel_elements = SUBARRAYS * SIMD_WIDTH;
            int repetition_factor = (parallel_elements > size)? 1 : ceil(size/parallel_elements);
            
            return repetition_factor*latency_simdram_relu[bit_precision];
        }
        else{
            int parallel_elements = 1 * SIMD_WIDTH;
            int repetition_factor = (parallel_elements > size)? 1 : ceil(size/parallel_elements);
            
            return repetition_factor*latency_simdram_relu[bit_precision];
        }
    }
    else{
        if (salp_enabled){
            int parallel_elements = SUBARRAYS * SIMD_WIDTH;
            int repetition_factor = (parallel_elements > size)? 1 : ceil(size/parallel_elements);
            
            return repetition_factor*latency_simdram_relu[32];
        }
        else{
            int parallel_elements = 1 * SIMD_WIDTH;
            int repetition_factor = (parallel_elements > size)? 1 : ceil(size/parallel_elements);
            
            return repetition_factor*latency_simdram_relu[32];
        }
    }
}


int get_simdram_adder_energy(int bit_precision, int size, bool dynamic_bit_precision_enabled, bool salp_enabled){
    if (dynamic_bit_precision_enabled){

        return 8*(bit_precision + 1)*ceil(size/SIMD_WIDTH)*AAP_ENERGY;
    }
    else{
        return 8*(32 + 1)*ceil(size/SIMD_WIDTH)*AAP_ENERGY;
    }
}

int get_relu_energy(int bit_precision, int size, bool dynamic_bit_precision_enabled, bool salp_enabled){
    if (dynamic_bit_precision_enabled){
        return (3*(bit_precision) + (bit_precision-1) % 2)*ceil(size/SIMD_WIDTH)*AAP_ENERGY;
    }
    else{
        return (3*(32) + (32-1) % 2)*ceil(size/SIMD_WIDTH)*AAP_ENERGY;
    }
}


int get_daftpum_adder_energy(int bit_precision, int size, char * adder_type, bool tfaw_enabled){
    // check if the adder_type array is empty
    // if so, we need to figure out the best adder type
    // Otherwise, we already know the adder type

    float full_adder_simdram_energy = 8*(32 + 1)*ceil(size/SIMD_WIDTH)*AAP_ENERGY;

    float full_adder_energy = 8.1075*bit_precision*ceil(size/SIMD_WIDTH)*AAP_ENERGY;

    float koggee_adder_energy = (0.025*bit_precision*bit_precision*bit_precision + 0.1*bit_precision*bit_precision + 5.5*log2(bit_precision)*log(bit_precision) - 5.5*log(bit_precision) +18.875*bit_precision -19)*AAP_ENERGY*ceil(size/SIMD_WIDTH);

    float brent_kung_adder_energy = (19.5*bit_precision - 10.8*log2(bit_precision) - 0.125)*AAP_ENERGY*ceil(size/SIMD_WIDTH);

    float ladner_fischer_adder_energy = (19.1*bit_precision + log2(bit_precision) - 19)*AAP_ENERGY*ceil(size/SIMD_WIDTH);

    float carry_select_adder_energy = 22.1465*bit_precision*AAP_ENERGY*ceil(size/SIMD_WIDTH);

    float rbr_adder_energy = 35.075*bit_precision*AAP_ENERGY*ceil(size/SIMD_WIDTH);

    int min_energy = full_adder_energy;

    if(strcmp(adder_type, "") == 0){

        bool found_smaller = false;
        // check the adder with the lowest energy
        if (koggee_adder_energy < min_energy){
            min_energy = koggee_adder_energy;
            strcpy(adder_type, "koggee_adder");
            found_smaller = true;
        }
        if (brent_kung_adder_energy < min_energy){
            min_energy = brent_kung_adder_energy;
            strcpy(adder_type, "brent_kung_adder");
            found_smaller = true;
        }
        if (ladner_fischer_adder_energy < min_energy){
            min_energy = ladner_fischer_adder_energy;
            strcpy(adder_type, "ladner_fischer_adder");
            found_smaller = true;
        }
        if (carry_select_adder_energy < min_energy){
            min_energy = carry_select_adder_energy;
            strcpy(adder_type, "carry_select_adder");
            found_smaller = true;
        }
        if (rbr_adder_energy < min_energy){
            min_energy = rbr_adder_energy;
            strcpy(adder_type, "rbr_adder");
            found_smaller = true;
        }

        if (full_adder_simdram_energy < min_energy){
            min_energy = full_adder_simdram_energy;
            strcpy(adder_type, "full_adder_simdram");
            found_smaller = true;
        }

        if (found_smaller == false){
            strcpy(adder_type, "full_adder");
        }
    }
    else{
        // get the energy of the adder type
        if (strcmp(adder_type, "full_adder") == 0){
            min_energy = full_adder_energy;
        }
        else if(strcmp(adder_type, "koggee_adder") == 0){
            min_energy = koggee_adder_energy;
        }
        else if(strcmp(adder_type, "brent_kung_adder") == 0){
            min_energy = brent_kung_adder_energy;
        }
        else if(strcmp(adder_type, "ladner_fischer_adder") == 0){
            min_energy = ladner_fischer_adder_energy;
        }
        else if(strcmp(adder_type, "carry_select_adder") == 0){
            min_energy = carry_select_adder_energy;
        }
        else if(strcmp(adder_type, "rbr_adder") == 0){
            min_energy = rbr_adder_energy;
        }
        else if (strcmp(adder_type, "full_adder_simdram") == 0){
            min_energy = full_adder_simdram_energy;
        }
        else{
            printf("Error: adder_type not recognized\n");
        }
    }

    return min_energy;
}


////////////////////////// Helper functions for multiplication //////////////////////////
int get_daftpum_multiplier_latency(int bit_precision, int size, char * adder_type, bool tfaw_enabled){
    int full_multiplier_latency = latency_simdram_ripple_carry_multiplier[bit_precision];
    int sklansky_multiplier_latency = latency_daftpum_sklansky_multiplier[bit_precision];
    int carryselect_multiplier_latency = latency_daftpum_csa_multiplier[bit_precision];
    int rbr_multiplier_latency = latency_daftpum_rbr_multiplier[bit_precision];

    // get the lowest latency
    int min_latency = full_multiplier_latency;

    // check if the adder_type array is empty
    // if so, we need to figure out the best adder type
    // Otherwise, we already know the adder type

     if(strcmp(adder_type, "") == 0){
        bool found_smaller = false;
        // check the adder with the lowest energy

        if (sklansky_multiplier_latency < min_latency){
            min_latency = sklansky_multiplier_latency;
            strcpy(adder_type, "sklansky_multiplier");
            //printf("sklansky_multiplier\n");

            found_smaller = true;
        }
        if (carryselect_multiplier_latency < min_latency){
            min_latency = carryselect_multiplier_latency;
            strcpy(adder_type, "carryselect_multiplier");

           // printf("carryselect_multiplier\n");

            found_smaller = true;

        }
        if (rbr_multiplier_latency < min_latency){
            min_latency = rbr_multiplier_latency;
            strcpy(adder_type, "rbr_multiplier");
            //printf("rbr_multiplier\n");

            found_smaller = true;
        }

        if (found_smaller == false){
            strcpy(adder_type, "full_multiplier");

            //printf("full_multiplier\n");
        }
    }
    else{
        // get the energy of the adder type
        if (strcmp(adder_type, "full_multiplier") == 0){
            min_latency = full_multiplier_latency;
        }
        else if(strcmp(adder_type, "sklansky_multiplier") == 0){
            min_latency = sklansky_multiplier_latency;
        }
        else if(strcmp(adder_type, "carryselect_multiplier") == 0){
            min_latency = carryselect_multiplier_latency;
        }
        else if(strcmp(adder_type, "rbr_multiplier") == 0){
            min_latency = rbr_multiplier_latency;
        }
        else{
            printf("Error: adder_type not recognized\n");
        }
    }

    if (tfaw_enabled == false){
        // get the closest power of two to the bit_precision
        //int closest_power_of_two = (int) pow(2, ceil(log2(bit_precision)));
        int closest_power_of_two = 1;
        
        int parallelism = SUBARRAYS/closest_power_of_two;

        int parallel_elements = parallelism * SIMD_WIDTH;

        int repetition_factor = (parallel_elements > size)? 1 : ceil(size/parallel_elements);

        min_latency = min_latency * repetition_factor;
    }
    else{
        // if number of bit_precision is larger than 4, there is no parallelism
        if (bit_precision >= 4){
            int parallel_elements = 1 * SIMD_WIDTH;
            int repetition_factor = (parallel_elements > size)? 1 : ceil(size/parallel_elements);
            min_latency = min_latency * repetition_factor;
        }
        else{
            //int closest_power_of_two = (int) pow(2, ceil(log2(bit_precision)));
            int closest_power_of_two = 1;
            int parallelism = 4/closest_power_of_two;
            int parallel_elements = parallelism * SIMD_WIDTH;
            int repetition_factor = (parallel_elements > size)? 1 : ceil(size/parallel_elements);
            min_latency = min_latency * repetition_factor;
        }
    }

    return min_latency;
}

int get_simdram_multiplier_latency(int bit_precision, int size, bool dynamic_bit_precision_enabled, bool salp_enabled){
    if (dynamic_bit_precision_enabled){

        if (salp_enabled){
            int parallel_elements = SUBARRAYS * SIMD_WIDTH;
            int repetition_factor = (parallel_elements > size)? 1 : ceil(size/parallel_elements);
            
            return repetition_factor*latency_simdram_ripple_carry_multiplier[bit_precision];
        }
        else{
            int parallel_elements = 1 * SIMD_WIDTH;
            int repetition_factor = (parallel_elements > size)? 1 : ceil(size/parallel_elements);
            
            return repetition_factor*latency_simdram_ripple_carry_multiplier[bit_precision];
        }
    }
    else{
        if (salp_enabled){
            int parallel_elements = SUBARRAYS * SIMD_WIDTH;
            int repetition_factor = (parallel_elements > size)? 1 : ceil(size/parallel_elements);
            
            if (bit_precision <= 32 ) return repetition_factor*latency_simdram_ripple_carry_multiplier[32];
            else return repetition_factor*latency_simdram_ripple_carry_multiplier[63]; 
        }
        else{
            int parallel_elements = 1 * SIMD_WIDTH;
            int repetition_factor = (parallel_elements > size)? 1 : ceil(size/parallel_elements);
            
            //printf("Repectition factor is %d\n", repetition_factor);

            if (bit_precision <= 32) return repetition_factor*latency_simdram_ripple_carry_multiplier[32];
            else return repetition_factor*latency_simdram_ripple_carry_multiplier[63];
        }
    }
}


int get_simdram_multiplier_energy(int bit_precision, int size, bool dynamic_bit_precision_enabled, bool salp_enabled){
    if (dynamic_bit_precision_enabled){

        return (11*bit_precision*bit_precision -5*bit_precision - 1)*ceil(size/SIMD_WIDTH)*AAP_ENERGY;
    }
    else{
        if (bit_precision <= 32 ) return (11*32*32 -5*bit_precision - 1)*ceil(size/SIMD_WIDTH)*AAP_ENERGY;
        else return (11*64*64 -5*bit_precision - 1)*ceil(size/SIMD_WIDTH)*AAP_ENERGY;
    }
}


int get_daftpum_multiplier_energy(int bit_precision, int size, char * adder_type, bool tfaw_enabled){
    // check if the adder_type array is empty
    // if so, we need to figure out the best adder type
    // Otherwise, we already know the adder type

    float full_multiplier_energy = (11*bit_precision*bit_precision -5*bit_precision - 1)*ceil(size/SIMD_WIDTH)*AAP_ENERGY;

    float sklansky_multiplier_energy = (4*bit_precision + 0.0075*bit_precision*(bit_precision-1) + 0.0075*2*0.1*bit_precision + bit_precision*(19.15*2*bit_precision + log2(2*bit_precision) -19))*ceil(size/SIMD_WIDTH)*AAP_ENERGY;

    float carry_select_multiplier_energy = (4*bit_precision + 0.0075*bit_precision*(bit_precision-1) + 0.0075*2*0.1*bit_precision + bit_precision*(19.15*2*bit_precision + log2(2*bit_precision) -19))*ceil(size/SIMD_WIDTH)*AAP_ENERGY;
    
    float rbr_adder_energy = (18.0325*bit_precision*bit_precision + 70.218*bit_precision)*ceil(size/SIMD_WIDTH)*AAP_ENERGY;

    int min_energy = full_multiplier_energy;

    if(strcmp(adder_type, "") == 0){

        bool found_smaller = false;
        // check the adder with the lowest energy
        if (sklansky_multiplier_energy < min_energy){
            min_energy = sklansky_multiplier_energy;
            strcpy(adder_type, "sklansky_multiplier");
            found_smaller = true;
        }
        if (carry_select_multiplier_energy < min_energy){
            min_energy = carry_select_multiplier_energy;
            strcpy(adder_type, "carryselect_multiplier");
            found_smaller = true;
        }
        if (rbr_adder_energy < min_energy){
            min_energy = rbr_adder_energy;
            strcpy(adder_type, "rbr_multiplier");
            found_smaller = true;
        }

        if (found_smaller == false){
            strcpy(adder_type, "full_multiplier");
        }
    }
    else{
        // get the energy of the adder type
        if (strcmp(adder_type, "full_multiplier") == 0){
            min_energy = full_multiplier_energy;
        }
        else if(strcmp(adder_type, "carryselect_multiplier") == 0){
            min_energy = carry_select_multiplier_energy;
        }
        else if(strcmp(adder_type, "sklansky_multiplier") == 0){
            min_energy = sklansky_multiplier_energy;
        }
        else if(strcmp(adder_type, "rbr_multiplier") == 0){
            min_energy = rbr_adder_energy;
        }
        else{
            printf("Error: adder_type not recognized\n");
        }
    }

    return min_energy;
}


///////////////////////////// DONE WITH HELPER FUNCTIONS FOR MULTIPLICATION /////////////////////////////

// Create an array to store the bbop_statistic
bbop_statistic bbop_statistics[MAX_BBOPS];

FILE *resolution = NULL;

unsigned long resolution_histogram[65]; 

void initialize_bbop_statistics(){
    for(int i = 0; i < MAX_BBOPS; i++){
        bbop_statistics[i].simdram_1_subarray_latency = 0.0;
        bbop_statistics[i].simdram_64_subarray_latency = 0.0;
        bbop_statistics[i].simdram_64_subarray_dynamic_precision_latency = 0.0;
        bbop_statistics[i].daftpum_static_latency_optimized_latency = 0.0;
        bbop_statistics[i].daftpum_static_energy_optimized_latency = 0.0;
        bbop_statistics[i].daftpum_latency_optimized_latency = 0.0;
        bbop_statistics[i].daftpum_energy_optimized_latency = 0.0;
        bbop_statistics[i].daftpum_tfaw_enabled_latency = 0.0;

        bbop_statistics[i].simdram_1_subarray_energy = 0.0;
        bbop_statistics[i].simdram_64_subarray_energy = 0.0;
        bbop_statistics[i].simdram_64_subarray_dynamic_precision_energy = 0.0;
        bbop_statistics[i].daftpum_static_latency_optimized_energy = 0.0;
        bbop_statistics[i].daftpum_static_energy_optimized_energy = 0.0;
        bbop_statistics[i].daftpum_latency_optimized_energy = 0.0;
        bbop_statistics[i].daftpum_energy_optimized_energy = 0.0;
        bbop_statistics[i].daftpum_tfaw_enabled_energy = 0.0;
        
        bbop_statistics[i].largest_input_a = 0;
        bbop_statistics[i].largest_input_b = 0;

        bbop_statistics[i].operation = -1;
    }

#ifdef PROFILE_RESOLUTION

    printf("Profiling resolution of the input elements of bbops\n");

    // create a file to store the resolution of the input elements of bbops
    resolution = fopen("bbop_resolution.csv", "w");

    //check if the file was created successfully
    if (resolution == NULL){
        printf("Error: Unable to create the file to store the resolution of the input elements of bbops\n");
        return;
    }

    // initialize the resolution histogram
    for (int i = 0; i < 65; i++){
        resolution_histogram[i] = 0;
    }

#endif 
}

int get_power_of_two_bit_precision(int bit_precision){
    // this function return the closest power of two to the bit_precision
    // available bit-precisions are 8, 16, 32, 64

    if (bit_precision <= 8){
        return 8;
    }
    else if (bit_precision <= 16){
        return 16;
    }
    else if (bit_precision <= 32){
        return 32;
    }
    else if (bit_precision <= 64){
        return 63;
    }
    else{
        return 63;
    }
}

void bbop_op(bbop_operation operation, DATATYPE_BBOP *A, DATATYPE_BBOP *B, DATATYPE_BBOP *C, unsigned long long size, int bbop_id, bbop_operation daftpum_operation){

    #ifdef PROFILE_RESOLUTION
        unsigned resolution_A;
        unsigned resolution_B;

        for (long i = 0; i < size; i++){
            resolution_A = (A[i] != 0) ? (unsigned int) floor(log2(abs(A[i]))) : 0;
            resolution_B = (B[i] != 0) ? (unsigned int) floor(log2(abs(B[i]))) : 0;

            // if resolution is larger than 64, set it to 64
            if (resolution_A > 64){
                resolution_A = 64;
            }
            if (resolution_B > 64){
                resolution_B = 64;
            }

            // check if he resolution is between 0 and 64
            if ((resolution_A >= 0) && (resolution_A <= 64)){
                resolution_histogram[resolution_A]++;
            } else{
                printf("Error: resolution_A is not between 0 and 64\n");
            }

            if ((resolution_B >= 0) && (resolution_B <= 64)){
                if (operation != BBOP_CPY) resolution_histogram[resolution_B]++;
            } else{
                printf("Error: resolution_B is not between 0 and 64\n");
            }
        }
    #endif 

    // Iterate through the array
    if ((operation == BBOP_ADD) || (operation == BBOP_ADD_8) || (operation == BBOP_ADD_16) || (operation == BBOP_ADD_32) || (operation == BBOP_ADD_64)){
        int largest_element = 0;
        int largest_element_a = 0;
        int largest_element_b    = 0;
        
        tvals maxinfo_a[NUM_THREADS];
        tvals maxinfo_b[NUM_THREADS];

        #pragma omp parallel shared(maxinfo_a, maxinfo_b)
        {
            
            int tid = omp_get_thread_num();
            maxinfo_a[tid].val = 0;
            maxinfo_b[tid].val = 0;

            #pragma omp for
            for (long i = 0; i < size; i++){
                C[i] = A[i] + B[i];
                if (INT4_DATA_TYPE == 1){

                    C[i] = (C[i] > INT4_MAX) ? INT4_MAX : C[i];
                    C[i] = (C[i] < INT4_MIN) ? INT4_MIN : C[i];
                }

                if (A[i] > maxinfo_a[tid].val){
                    maxinfo_a[tid].val = A[i];
                }

                if (B[i] > maxinfo_b[tid].val){
                    maxinfo_b[tid].val = B[i];
                }
            }

            #pragma omp flush(maxinfo_a, maxinfo_b)
            #pragma omp master
            {
                int nt = omp_get_num_threads();
                largest_element_a = 0;
                largest_element_b = 0;

                for(int i = 0; i < nt; i++){
                    if (maxinfo_a[i].val > largest_element_a){
                        largest_element_a = maxinfo_a[i].val;
                    }

                    if (maxinfo_b[i].val > largest_element_b){
                        largest_element_b = maxinfo_b[i].val;
                    }
                }
            }
        }

        largest_element = (largest_element_a > largest_element_b) ? largest_element_a : largest_element_b;

        // check for the largest input
        if (largest_element_a > bbop_statistics[bbop_id].largest_input_a){
            bbop_statistics[bbop_id].largest_input_a = largest_element_a;
        }

        if (largest_element_b > bbop_statistics[bbop_id].largest_input_b){
            bbop_statistics[bbop_id].largest_input_b = largest_element_b;
        }

        // calculate the bit_precision based on the largest element 
        int bit_precision = (int) floor(log2(largest_element)) + 1;
        
        // create an empty char * to store the adder type
        char adder_type[40];
        // set the adder_type to empty
        strcpy(adder_type, "");

        
        if (bit_precision <= 0){
            bit_precision = 1;
        }

        if (bit_precision >= 64){
            bit_precision = 63;
        }
        
        int static_bit_precision = get_power_of_two_bit_precision(bit_precision);
        
        //int get_simdram_adder_latency(int bit_precision, int size, bool dynamic_bit_precision_enabled, bool salp_enabled){
        bbop_statistics[bbop_id].simdram_1_subarray_latency += get_simdram_adder_latency(bit_precision, size, false, false);
        bbop_statistics[bbop_id].simdram_64_subarray_latency += get_simdram_adder_latency(bit_precision, size, false, true);
        bbop_statistics[bbop_id].simdram_64_subarray_dynamic_precision_latency += get_simdram_adder_latency(bit_precision, size, true, true);
        bbop_statistics[bbop_id].simdram_1_subarray_energy += get_simdram_adder_energy(bit_precision, size, false, false);
        bbop_statistics[bbop_id].simdram_64_subarray_energy += get_simdram_adder_energy(bit_precision, size, false, true);
        bbop_statistics[bbop_id].simdram_64_subarray_dynamic_precision_energy += get_simdram_adder_energy(bit_precision, size, true, true);


        bbop_statistics[bbop_id].daftpum_static_latency_optimized_latency += get_daftpum_adder_latency(static_bit_precision, size, adder_type, false);
        bbop_statistics[bbop_id].daftpum_static_latency_optimized_energy += get_daftpum_adder_energy(static_bit_precision, size, adder_type, false);

        // make the adder_type empty again
        strcpy(adder_type, "");
        bbop_statistics[bbop_id].daftpum_latency_optimized_latency += get_daftpum_adder_latency(bit_precision, size, adder_type, false);
        bbop_statistics[bbop_id].daftpum_tfaw_enabled_latency += get_daftpum_adder_latency(bit_precision, size, adder_type, true);
        bbop_statistics[bbop_id].daftpum_latency_optimized_energy += get_daftpum_adder_energy(bit_precision, size, adder_type, false);
        bbop_statistics[bbop_id].daftpum_tfaw_enabled_energy += get_daftpum_adder_energy(bit_precision, size, adder_type, true);


        // make the adder_type empty again
        strcpy(adder_type, "");
        bbop_statistics[bbop_id].daftpum_static_energy_optimized_energy += get_daftpum_adder_energy(static_bit_precision, size, adder_type, false);
        bbop_statistics[bbop_id].daftpum_static_energy_optimized_latency += get_daftpum_adder_latency(static_bit_precision, size, adder_type, false);

        // make the adder_type empty again
        strcpy(adder_type, "");
        bbop_statistics[bbop_id].daftpum_energy_optimized_energy += get_daftpum_adder_energy(bit_precision, size, adder_type, false);
        bbop_statistics[bbop_id].daftpum_energy_optimized_latency += get_daftpum_adder_latency(bit_precision, size, adder_type, false);
        
        bbop_statistics[bbop_id].operation = operation;
    }
    else if ((operation == BBOP_SUB) || (operation == BBOP_SUB_8) || (operation == BBOP_SUB_16) || (operation == BBOP_SUB_32) || (operation == BBOP_SUB_64)){

        int largest_element = 0;
        int largest_element_a = 0;
        int largest_element_b    = 0;
        
        tvals maxinfo_a[NUM_THREADS];
        tvals maxinfo_b[NUM_THREADS];

        #pragma omp parallel shared(maxinfo_a, maxinfo_b)
        {
            
            int tid = omp_get_thread_num();
            maxinfo_a[tid].val = 0;
            maxinfo_b[tid].val = 0;

            #pragma omp for
            for (long i = 0; i < size; i++){
                C[i] = A[i] - B[i];
                if (INT4_DATA_TYPE == 1){
                    C[i] = (C[i] > INT4_MAX) ? INT4_MAX : C[i];
                    C[i] = (C[i] < INT4_MIN) ? INT4_MIN : C[i];
                }

                if (A[i] > maxinfo_a[tid].val){
                    maxinfo_a[tid].val = A[i];
                }

                if (B[i] > maxinfo_b[tid].val){
                    maxinfo_b[tid].val = B[i];
                }
            }

            #pragma omp flush(maxinfo_a, maxinfo_b)
            #pragma omp master
            {
                int nt = omp_get_num_threads();
                largest_element_a = 0;
                largest_element_b = 0;

                for(int i = 0; i < nt; i++){
                    if (maxinfo_a[i].val > largest_element_a){
                        largest_element_a = maxinfo_a[i].val;
                    }

                    if (maxinfo_b[i].val > largest_element_b){
                        largest_element_b = maxinfo_b[i].val;
                    }
                }
            }
        }

        largest_element = (largest_element_a > largest_element_b) ? largest_element_a : largest_element_b;

        // check for the largest input
        if (largest_element_a > bbop_statistics[bbop_id].largest_input_a){
            bbop_statistics[bbop_id].largest_input_a = largest_element_a;
        }

        if (largest_element_b > bbop_statistics[bbop_id].largest_input_b){
            bbop_statistics[bbop_id].largest_input_b = largest_element_b;
        }

        // calculate the bit_precision based on the largest element 
        int bit_precision = (int) floor(log2(largest_element)) + 1;

        if (bit_precision <= 0){
            bit_precision = 1;
        }

        if (bit_precision >= 64){
            bit_precision = 63;
        }
        
        // create an empty char * to store the adder type
        char adder_type[40];
        // set the adder_type to empty
        strcpy(adder_type, "");

        int static_bit_precision = get_power_of_two_bit_precision(bit_precision);
        
        //int get_simdram_adder_latency(int bit_precision, int size, bool dynamic_bit_precision_enabled, bool salp_enabled){
        bbop_statistics[bbop_id].simdram_1_subarray_latency += get_simdram_adder_latency(bit_precision, size, false, false);
        bbop_statistics[bbop_id].simdram_64_subarray_latency += get_simdram_adder_latency(bit_precision, size, false, true);
        bbop_statistics[bbop_id].simdram_64_subarray_dynamic_precision_latency += get_simdram_adder_latency(bit_precision, size, true, true);
        bbop_statistics[bbop_id].simdram_1_subarray_energy += get_simdram_adder_energy(bit_precision, size, false, false);
        bbop_statistics[bbop_id].simdram_64_subarray_energy += get_simdram_adder_energy(bit_precision, size, false, true);
        bbop_statistics[bbop_id].simdram_64_subarray_dynamic_precision_energy += get_simdram_adder_energy(bit_precision, size, true, true);


        bbop_statistics[bbop_id].daftpum_static_latency_optimized_latency += get_daftpum_adder_latency(static_bit_precision, size, adder_type, false);
        bbop_statistics[bbop_id].daftpum_static_latency_optimized_energy += get_daftpum_adder_energy(static_bit_precision, size, adder_type, false);

        // make the adder_type empty again
        strcpy(adder_type, "");
        bbop_statistics[bbop_id].daftpum_latency_optimized_latency += get_daftpum_adder_latency(bit_precision, size, adder_type, false);
        bbop_statistics[bbop_id].daftpum_tfaw_enabled_latency += get_daftpum_adder_latency(bit_precision, size, adder_type, true);
        bbop_statistics[bbop_id].daftpum_latency_optimized_energy += get_daftpum_adder_energy(bit_precision, size, adder_type, false);
        bbop_statistics[bbop_id].daftpum_tfaw_enabled_energy += get_daftpum_adder_energy(bit_precision, size, adder_type, true);


        // make the adder_type empty again
        strcpy(adder_type, "");
        bbop_statistics[bbop_id].daftpum_static_energy_optimized_energy += get_daftpum_adder_energy(static_bit_precision, size, adder_type, false);
        bbop_statistics[bbop_id].daftpum_static_energy_optimized_latency += get_daftpum_adder_latency(static_bit_precision, size, adder_type, false);

        // make the adder_type empty again
        strcpy(adder_type, "");
        bbop_statistics[bbop_id].daftpum_energy_optimized_energy += get_daftpum_adder_energy(bit_precision, size, adder_type, false);
        bbop_statistics[bbop_id].daftpum_energy_optimized_latency += get_daftpum_adder_latency(bit_precision, size, adder_type, false);
        
        bbop_statistics[bbop_id].operation = operation;
    }
    else if ((operation == BBOP_MUL) || (operation == BBOP_MUL_8) || (operation == BBOP_MUL_16) || (operation == BBOP_MUL_32) || (operation == BBOP_MUL_64)){

        int largest_element = 0;
        int largest_element_a = 0;
        int largest_element_b    = 0;
        
        tvals maxinfo_a[NUM_THREADS];
        tvals maxinfo_b[NUM_THREADS];

        #pragma omp parallel shared(maxinfo_a, maxinfo_b)
        {
            
            int tid = omp_get_thread_num();
            maxinfo_a[tid].val = 0;
            maxinfo_b[tid].val = 0;

            #pragma omp for
            for (unsigned long long i = 0; i < size; i++){
                if (i >= size){
                    printf("[DEBUG] Error: Index out of bounds\n");
                }

                C[i] = (DATATYPE_BBOP) A[i] * B[i];
                if (INT4_DATA_TYPE == 1){
                    C[i] = (C[i] > INT4_MAX) ? INT4_MAX : C[i];
                    C[i] = (C[i] < INT4_MIN) ? INT4_MIN : C[i];
                }
                // printf("C[%d] = A[%d] * B[%d] = %d * %d = %d\n", i, i, i, A[i], B[i], C[i]);
                
                if (A[i] > maxinfo_a[tid].val){
                    maxinfo_a[tid].val = A[i];
                }

                if (B[i] > maxinfo_b[tid].val){
                    maxinfo_b[tid].val = B[i];
                }
            }

            #pragma omp flush(maxinfo_a, maxinfo_b)
            #pragma omp master
            {
                int nt = omp_get_num_threads();
                largest_element_a = 0;
                largest_element_b = 0;

                for(int i = 0; i < nt; i++){
                    if (maxinfo_a[i].val > largest_element_a){
                        largest_element_a = maxinfo_a[i].val;
                    }

                    if (maxinfo_b[i].val > largest_element_b){
                        largest_element_b = maxinfo_b[i].val;
                    }
                }
            }
        }

        largest_element = (largest_element_a > largest_element_b) ? largest_element_a : largest_element_b;
        
        // check for the largest input
        if (largest_element_a > bbop_statistics[bbop_id].largest_input_a){
            bbop_statistics[bbop_id].largest_input_a = largest_element_a;
        }

        if (largest_element_b > bbop_statistics[bbop_id].largest_input_b){
            bbop_statistics[bbop_id].largest_input_b = largest_element_b;
        }

        // calculate the bit_precision based on the largest element 
        int bit_precision = (int) floor(log2(largest_element)) + 1;

        if (bit_precision <= 0){
            bit_precision = 1;
        }

        if (bit_precision >= 64){
            bit_precision = 63;
        }
        
        // create an empty char * to store the adder type
        char adder_type[40];
        // set the adder_type to empty
        strcpy(adder_type, "");
        
        int static_bit_precision = get_power_of_two_bit_precision(bit_precision);

        //int get_simdram_adder_latency(int bit_precision, int size, bool dynamic_bit_precision_enabled, bool salp_enabled){
        bbop_statistics[bbop_id].simdram_1_subarray_latency += get_simdram_multiplier_latency(bit_precision, size, false, false);
        bbop_statistics[bbop_id].simdram_64_subarray_latency += get_simdram_multiplier_latency(bit_precision, size, false, true);
        bbop_statistics[bbop_id].simdram_64_subarray_dynamic_precision_latency += get_simdram_multiplier_latency(bit_precision, size, true, true);
        bbop_statistics[bbop_id].simdram_1_subarray_energy += get_simdram_multiplier_energy(bit_precision, size, false, false);
        bbop_statistics[bbop_id].simdram_64_subarray_energy += get_simdram_multiplier_energy(bit_precision, size, false, true);
        bbop_statistics[bbop_id].simdram_64_subarray_dynamic_precision_energy += get_simdram_multiplier_energy(bit_precision, size, true, true);


        bbop_statistics[bbop_id].daftpum_static_latency_optimized_latency += get_daftpum_multiplier_latency(static_bit_precision, size, adder_type, false);
        bbop_statistics[bbop_id].daftpum_static_latency_optimized_energy += get_daftpum_multiplier_energy(static_bit_precision, size, adder_type, false);

        // make the adder_type empty again
        strcpy(adder_type, "");
        bbop_statistics[bbop_id].daftpum_latency_optimized_latency += get_daftpum_multiplier_latency(bit_precision, size, adder_type, false);
        bbop_statistics[bbop_id].daftpum_tfaw_enabled_latency += get_daftpum_multiplier_latency(bit_precision, size, adder_type, true);
        bbop_statistics[bbop_id].daftpum_latency_optimized_energy += get_daftpum_multiplier_energy(bit_precision, size, adder_type, false);
        bbop_statistics[bbop_id].daftpum_tfaw_enabled_energy += get_daftpum_multiplier_energy(bit_precision, size, adder_type, true);

        // make the adder_type empty again
        strcpy(adder_type, "");
        bbop_statistics[bbop_id].daftpum_static_energy_optimized_energy += get_daftpum_multiplier_energy(static_bit_precision, size, adder_type, false);
        bbop_statistics[bbop_id].daftpum_static_energy_optimized_latency += get_daftpum_multiplier_latency(static_bit_precision, size, adder_type, false);

        // make the adder_type empty again
        strcpy(adder_type, "");
        bbop_statistics[bbop_id].daftpum_energy_optimized_energy += get_daftpum_multiplier_energy(bit_precision, size, adder_type, false);
        bbop_statistics[bbop_id].daftpum_energy_optimized_latency += get_daftpum_multiplier_latency(bit_precision, size, adder_type, false);
        
	    bbop_statistics[bbop_id].operation = operation;

    }
    else if ((operation == BBOP_DIV) || (operation == BBOP_DIV_8) || (operation == BBOP_DIV_16) || (operation == BBOP_DIV_32) || (operation == BBOP_DIV_64)){

        int largest_element = 0;
        int largest_element_a = 0;
        int largest_element_b    = 0;
        
        tvals maxinfo_a[NUM_THREADS];
        tvals maxinfo_b[NUM_THREADS];


        #pragma omp parallel shared(maxinfo_a, maxinfo_b)
        {
            
            int tid = omp_get_thread_num();
            maxinfo_a[tid].val = 0;
            maxinfo_b[tid].val = 0;

            #pragma omp for
            for (long i = 0; i < size; i++){
                C[i] = (B[i] > 0) ? A[i] + B[i] : 0;
                if (INT4_DATA_TYPE == 1){
                    C[i] = (C[i] > INT4_MAX) ? INT4_MAX : C[i];
                    C[i] = (C[i] < INT4_MIN) ? INT4_MIN : C[i];
                }

                if (A[i] > maxinfo_a[tid].val){
                    maxinfo_a[tid].val = A[i];
                }

                if (B[i] > maxinfo_b[tid].val){
                    maxinfo_b[tid].val = B[i];
                }
            }

            #pragma omp flush(maxinfo_a, maxinfo_b)
            #pragma omp master
            {
                int nt = omp_get_num_threads();
                largest_element_a = 0;
                largest_element_b = 0;

                for(int i = 0; i < nt; i++){
                    if (maxinfo_a[i].val > largest_element_a){
                        largest_element_a = maxinfo_a[i].val;
                    }

                    if (maxinfo_b[i].val > largest_element_b){
                        largest_element_b = maxinfo_b[i].val;
                    }
                }
            }
        }

        largest_element = (largest_element_a > largest_element_b) ? largest_element_a : largest_element_b;

        // check for the largest input
        if (largest_element_a > bbop_statistics[bbop_id].largest_input_a){
            bbop_statistics[bbop_id].largest_input_a = largest_element_a;
        }

        if (largest_element_b > bbop_statistics[bbop_id].largest_input_b){
            bbop_statistics[bbop_id].largest_input_b = largest_element_b;
        }

        // calculate the bit_precision based on the largest element 
        int bit_precision = (int) floor(log2(largest_element)) + 1;

        if (bit_precision <= 0){
            bit_precision = 1;
        }

        if (bit_precision >= 64){
            bit_precision = 63;
        }

        // print the bit_precision
        // printf("[DIV] bit_precision is %d\n", bit_precision);
        
        // create an empty char * to store the adder type
        char adder_type[40];
        // set the adder_type to empty
        strcpy(adder_type, "");

        
        int static_bit_precision = get_power_of_two_bit_precision(bit_precision);
        
        //int get_simdram_adder_latency(int bit_precision, int size, bool dynamic_bit_precision_enabled, bool salp_enabled){
        bbop_statistics[bbop_id].simdram_1_subarray_latency += get_simdram_multiplier_latency(bit_precision, size, false, false);
        bbop_statistics[bbop_id].simdram_64_subarray_latency += get_simdram_multiplier_latency(bit_precision, size, false, true);
        bbop_statistics[bbop_id].simdram_64_subarray_dynamic_precision_latency += get_simdram_multiplier_latency(bit_precision, size, true, true);
        bbop_statistics[bbop_id].simdram_1_subarray_energy += get_simdram_multiplier_energy(bit_precision, size, false, false);
        bbop_statistics[bbop_id].simdram_64_subarray_energy += get_simdram_multiplier_energy(bit_precision, size, false, true);
        bbop_statistics[bbop_id].simdram_64_subarray_dynamic_precision_energy += get_simdram_multiplier_energy(bit_precision, size, true, true);


        bbop_statistics[bbop_id].daftpum_static_latency_optimized_latency += get_daftpum_multiplier_latency(static_bit_precision, size, adder_type, false);
        bbop_statistics[bbop_id].daftpum_static_latency_optimized_energy += get_daftpum_multiplier_energy(static_bit_precision, size, adder_type, false);

        // make the adder_type empty again
        strcpy(adder_type, "");
        bbop_statistics[bbop_id].daftpum_latency_optimized_latency += get_daftpum_multiplier_latency(bit_precision, size, adder_type, false);
        bbop_statistics[bbop_id].daftpum_tfaw_enabled_latency += get_daftpum_multiplier_latency(bit_precision, size, adder_type, true);
        bbop_statistics[bbop_id].daftpum_latency_optimized_energy += get_daftpum_multiplier_energy(bit_precision, size, adder_type, false);
        bbop_statistics[bbop_id].daftpum_tfaw_enabled_energy += get_daftpum_multiplier_energy(bit_precision, size, adder_type, true);

        // make the adder_type empty again
        strcpy(adder_type, "");
        bbop_statistics[bbop_id].daftpum_static_energy_optimized_energy += get_daftpum_multiplier_energy(static_bit_precision, size, adder_type, false);
        bbop_statistics[bbop_id].daftpum_static_energy_optimized_latency += get_daftpum_multiplier_latency(static_bit_precision, size, adder_type, false);

        // make the adder_type empty again
        strcpy(adder_type, "");
        bbop_statistics[bbop_id].daftpum_energy_optimized_energy += get_daftpum_multiplier_energy(bit_precision, size, adder_type, false);
        bbop_statistics[bbop_id].daftpum_energy_optimized_latency += get_daftpum_multiplier_latency(bit_precision, size, adder_type, false);
        
        bbop_statistics[bbop_id].operation = operation;
    }
    else if (operation == BBOP_CPY){

        int largest_element = 0;
        int largest_element_a = 0;
        
        tvals maxinfo_a[NUM_THREADS];

        #pragma omp parallel shared(maxinfo_a)
        {
            
            int tid = omp_get_thread_num();
            maxinfo_a[tid].val = 0;
        
            #pragma omp for
            for (long i = 0; i < size; i++){
                C[i] = A[i];
                if (INT4_DATA_TYPE == 1){
                    C[i] = (C[i] > INT4_MAX) ? INT4_MAX : C[i];
                    C[i] = (C[i] < INT4_MIN) ? INT4_MIN : C[i];
                }

                if (A[i] > maxinfo_a[tid].val){
                    maxinfo_a[tid].val = A[i];
                }
            }

            #pragma omp flush(maxinfo_a)
            #pragma omp master
            {
                int nt = omp_get_num_threads();
                largest_element_a = 0;

                for(int i = 0; i < nt; i++){
                    if (maxinfo_a[i].val > largest_element_a){
                        largest_element_a = maxinfo_a[i].val;
                    }
                }
            }
        }

        largest_element = largest_element_a;

        // check for the largest input
        if (largest_element_a > bbop_statistics[bbop_id].largest_input_a){
            bbop_statistics[bbop_id].largest_input_a = largest_element_a;
        }

        // calculate the bit_precision based on the largest element 
        int bit_precision = (int) floor(log2(largest_element)) + 1;

        if (bit_precision <= 0){
            bit_precision = 1;
        }

        if (bit_precision >= 64){
            bit_precision = 63;
        }

        int static_bit_precision = get_power_of_two_bit_precision(bit_precision);

        //int get_simdram_adder_latency(int bit_precision, int size, bool dynamic_bit_precision_enabled, bool salp_enabled){
        bbop_statistics[bbop_id].simdram_1_subarray_latency += get_relu_latency(bit_precision, size, false, false);
        bbop_statistics[bbop_id].simdram_64_subarray_latency += get_relu_latency(bit_precision, size, false, true);
        bbop_statistics[bbop_id].simdram_64_subarray_dynamic_precision_latency += get_relu_latency(bit_precision, size, true, true);
        bbop_statistics[bbop_id].simdram_1_subarray_energy += get_relu_energy(bit_precision, size, false, false);
        bbop_statistics[bbop_id].simdram_64_subarray_energy += get_relu_energy(bit_precision, size, false, true);
        bbop_statistics[bbop_id].simdram_64_subarray_dynamic_precision_energy += get_relu_energy(bit_precision, size, true, true);


        bbop_statistics[bbop_id].daftpum_static_latency_optimized_latency += get_relu_latency(static_bit_precision, size, true, true);
        bbop_statistics[bbop_id].daftpum_static_latency_optimized_energy += get_relu_energy(static_bit_precision, size, true, true);

 
        bbop_statistics[bbop_id].daftpum_latency_optimized_latency += get_relu_latency(bit_precision, size, true, true);
        bbop_statistics[bbop_id].daftpum_tfaw_enabled_latency += get_relu_latency(bit_precision, size, true, true);
        bbop_statistics[bbop_id].daftpum_latency_optimized_energy += get_relu_energy(bit_precision, size, true, true);
        bbop_statistics[bbop_id].daftpum_tfaw_enabled_energy += get_relu_energy(bit_precision, size, true, true);

        bbop_statistics[bbop_id].daftpum_static_energy_optimized_energy += get_relu_energy(static_bit_precision, size, true, true);
        bbop_statistics[bbop_id].daftpum_static_energy_optimized_latency += get_relu_latency(static_bit_precision, size, true, true);

        bbop_statistics[bbop_id].daftpum_energy_optimized_energy += get_relu_energy(bit_precision, size, true, true);
        bbop_statistics[bbop_id].daftpum_energy_optimized_latency +=  get_relu_latency(bit_precision, size, true, true);
        
        bbop_statistics[bbop_id].operation = operation;
    }
     else if (operation == BBOP_RELU){
        int largest_element = 0;
        int largest_element_a = 0;
        
        tvals maxinfo_a[NUM_THREADS];

        #pragma omp parallel shared(maxinfo_a)
        {
            
            int tid = omp_get_thread_num();
            maxinfo_a[tid].val = 0;
        
            #pragma omp for
            for (long i = 0; i < size; i++){
                C[i] = mmax(0, A[i]);
                if (INT4_DATA_TYPE == 1){
                    C[i] = (C[i] > INT4_MAX) ? INT4_MAX : C[i];
                    C[i] = (C[i] < INT4_MIN) ? INT4_MIN : C[i];
                }

                if (A[i] > maxinfo_a[tid].val){
                    maxinfo_a[tid].val = A[i];
                }
            }

            #pragma omp flush(maxinfo_a)
            #pragma omp master
            {
                int nt = omp_get_num_threads();
                largest_element_a = 0;

                for(int i = 0; i < nt; i++){
                    if (maxinfo_a[i].val > largest_element_a){
                        largest_element_a = maxinfo_a[i].val;
                    }
                }
            }
        }

        largest_element = largest_element_a;

        // check for the largest input
        if (largest_element_a > bbop_statistics[bbop_id].largest_input_a){
            bbop_statistics[bbop_id].largest_input_a = largest_element_a;
        }

        // calculate the bit_precision based on the largest element 
        int bit_precision = (int) floor(log2(largest_element)) + 1;

        if (bit_precision <= 0){
            bit_precision = 1;
        }

        if (bit_precision >= 64){
            bit_precision = 63;
        }

        int static_bit_precision = get_power_of_two_bit_precision(bit_precision);

        bbop_statistics[bbop_id].simdram_1_subarray_latency += 32*ceil(size/SIMD_WIDTH)*AAP_LATENCY;
        bbop_statistics[bbop_id].simdram_64_subarray_latency += 32*ceil(size/(SIMD_WIDTH*64))*AAP_LATENCY;
        bbop_statistics[bbop_id].simdram_64_subarray_dynamic_precision_latency += bit_precision*ceil(size/(SIMD_WIDTH*64))*AAP_LATENCY;

        bbop_statistics[bbop_id].daftpum_static_latency_optimized_latency += static_bit_precision*ceil(size/(SIMD_WIDTH*64))*AAP_LATENCY;
        bbop_statistics[bbop_id].daftpum_latency_optimized_latency += bit_precision*ceil(size/(SIMD_WIDTH*64))*AAP_LATENCY;
        bbop_statistics[bbop_id].daftpum_tfaw_enabled_latency += bit_precision*ceil(size/(SIMD_WIDTH*4))*AAP_LATENCY;

        bbop_statistics[bbop_id].simdram_1_subarray_energy += 32*ceil(size/SIMD_WIDTH)*AAP_ENERGY;
        bbop_statistics[bbop_id].simdram_64_subarray_energy += 32*ceil(size/SIMD_WIDTH)*AAP_ENERGY;
        bbop_statistics[bbop_id].simdram_64_subarray_dynamic_precision_energy += bit_precision*ceil(size/SIMD_WIDTH)*AAP_ENERGY;

        bbop_statistics[bbop_id].daftpum_static_latency_optimized_energy += static_bit_precision*ceil(size/SIMD_WIDTH)*AAP_ENERGY;
        bbop_statistics[bbop_id].daftpum_latency_optimized_energy += bit_precision*ceil(size/SIMD_WIDTH)*AAP_ENERGY;
        bbop_statistics[bbop_id].daftpum_tfaw_enabled_energy += bit_precision*ceil(size/SIMD_WIDTH)*AAP_ENERGY;

        // make the adder_type empty again
        bbop_statistics[bbop_id].daftpum_static_energy_optimized_energy += static_bit_precision*ceil(size/SIMD_WIDTH)*AAP_ENERGY;
        bbop_statistics[bbop_id].daftpum_static_energy_optimized_latency += static_bit_precision*ceil(size/(SIMD_WIDTH*64))*AAP_LATENCY;

        bbop_statistics[bbop_id].daftpum_energy_optimized_energy += bit_precision*ceil(size/SIMD_WIDTH)*AAP_ENERGY;
        bbop_statistics[bbop_id].daftpum_energy_optimized_latency += bit_precision*ceil(size/(SIMD_WIDTH*64))*AAP_LATENCY;
        bbop_statistics[bbop_id].operation = operation;
    }
    else if (operation == CPU_CPY){
        for (long i = 0; i < size; i++){
            C[i] = A[i];
        }
    }
    else{
        printf("Error: bbop_operation not recognized\n");
    }

    // Check if loop_id is valid
    if (bbop_id >= MAX_BBOPS){
        printf("Error: bbop_id is greater than MAX_BBOPS");
        return;
    }
    
}


DATATYPE_BBOP bbop_op_red(bbop_operation operation, DATATYPE_BBOP *A, long size, int bbop_id){

    #ifdef PROFILE_RESOLUTION
        unsigned resolution_A;
        for (long i = 0; i < size; i++){
            resolution_A = (A[i] != 0) ? (unsigned int) floor(log2(abs(A[i]))) : 0;

            // if resolution is larger than 64, set it to 64
            if (resolution_A > 64){
                resolution_A = 64;
            }

            // check if he resolution is between 0 and 64
            if ((resolution_A >= 0) && (resolution_A <= 64)){
                resolution_histogram[resolution_A]++;
            } else{
                printf("Error: resolution_A is not between 0 and 64\n");
            }
        }
    #endif 

    DATATYPE_BBOP output = 0;
    for (int i = 0; i < size; i++){
        switch(operation){
            case BBOP_ADD:
                output += A[i];
                // printf("Adding A[%d]=%ld, output=%ld\n", i, A[i], output);
                break;
            case BBOP_SUB:
                output -= A[i];
                break;
            case BBOP_MUL:
                output *= A[i];
                break;
            case BBOP_DIV:
                output /= A[i];
                break;
            case BBOP_ADD_8:
                output += A[i];
                break;
            default:
                printf("Error: Invalid operation");
                return -1;
        }
        if (INT4_DATA_TYPE == 1){
            output = (output > INT4_MAX) ? INT4_MAX : output;
            output = (output < INT4_MIN) ? INT4_MIN : output;
        }

        // check for the largest input
        if (A[i] > bbop_statistics[bbop_id].largest_input_a){
            bbop_statistics[bbop_id].largest_input_a = A[i];
        }

        if (output > bbop_statistics[bbop_id].largest_input_b){
            bbop_statistics[bbop_id].largest_input_b = output;
        }
    }
    // printf("%d\n", output);

    // if(output!=0) printf("output is %ld\n", output);
    
    // Update the operation
    bbop_statistics[bbop_id].operation = BBOP_RED; 

    
    // Check if loop_id is valid
    if (bbop_id >= MAX_BBOPS){
        printf("Error: bbop_id is greater than MAX_BBOPS");
        return -1;
    }

    return output;
}

char * get_bbop_name(bbop_operation bbop_op){
    switch (bbop_op){
        case BBOP_ADD:
            return "BBOP_ADD";
        case BBOP_ADD_8:
            return "BBOP_ADD_8";
        case BBOP_ADD_16:
            return "BBOP_ADD_16";
        case BBOP_ADD_32:
            return "BBOP_ADD_32";
        case BBOP_ADD_64:
            return "BBOP_ADD_64";
        case BBOP_SUB:
            return "BBOP_SUB";
        case BBOP_SUB_8:
            return "BBOP_SUB_8";
        case BBOP_SUB_16:
            return "BBOP_SUB_16";
        case BBOP_SUB_32:
            return "BBOP_SUB_32";
        case BBOP_SUB_64:
            return "BBOP_SUB_64";    
        case BBOP_MUL:
            return "BBOP_MUL";
        case BBOP_MUL_8:
            return "BBOP_MUL_8";
        case BBOP_MUL_16:
            return "BBOP_MUL_16";
        case BBOP_MUL_32:
            return "BBOP_MUL_32";
        case BBOP_MUL_64:
            return "BBOP_MUL_64";
        case BBOP_DIV:
            return "BBOP_DIV";
        case BBOP_DIV_8:
            return "BBOP_DIV_8";
        case BBOP_DIV_16:
            return "BBOP_DIV_16";
        case BBOP_DIV_32:
            return "BBOP_DIV_32";
        case BBOP_DIV_64:
            return "BBOP_DIV_64";
        case BBOP_CPY:
            return "BBOP_CPY";
        case BBOP_RELU:
            return "BBOP_RELU";
        case BBOP_RED:
            return "BBOP_REDUCE";
        case CPU_CPY:
            return "CPU_CPY";
        default:
            return "BBOP_UNKNOWN";
    }
}

void print_bbop_statistic(){

    printf("[DEBUG] Printing the statistics of the BBOPs\n");
    char output_file_name[100];

    if (OUT_APPEND != 0){
        // concatenate the output file name stored in OUT_NAME with the file extension ".csv"
        sprintf(output_file_name, "bbop_statistics_%d.csv", OUT_APPEND);
    }
    else{
        // concatenate the output file name stored in OUT_NAME with the file extension ".csv"
        if (INT4_DATA_TYPE == 1)
            sprintf(output_file_name, "bbop_statistics_int4.csv");
        else
            sprintf(output_file_name, "bbop_statistics.csv");
    }
    
    // Create a file to store the statistics of the BBOPs using fopen
    FILE *fp = fopen(output_file_name, "w");

    // Write the header of the file, which should be: 
    // bbop_id, Mechanism, latency, energy, avg_utilization, min_utilization, max_utilization
    fprintf(fp, "bbop_id, operation, largest input A, largest input (red. output) B, mechanism, latency (ms), energy (mJ)\n");

    // Summary statistics 
    double simdram_1_subarray_latency_total = 0;
    double simdram_64_subarray_latency_total = 0;
    double simdram_64_subarray_dynamic_precision_latency_total = 0;
    double daftpum_static_latency_optimized_latency_total = 0;
    double daftpum_static_energy_optimized_latency_total = 0;
    double daftpum_latency_optimized_latency_total = 0;
    double daftpum_tfaw_enabled_latency_total = 0;
    double daftpum_energy_optimized_latency_total = 0;

    double simdram_1_subarray_energy_total = 0;
    double simdram_64_subarray_energy_total = 0;
    double simdram_64_subarray_dynamic_precision_energy_total = 0;
    double daftpum_static_latency_optimized_energy_total = 0;
    double daftpum_static_energy_optimized_energy_total = 0;
    double daftpum_latency_optimized_energy_total = 0;
    double daftpum_tfaw_enabled_energy_total = 0;
    double daftpum_energy_optimized_energy_total = 0;

    int bbop_count = 0;

    // Iterate over all the bbops 
    
    for (int i = 0; i < MAX_BBOPS; i++){
        if (bbop_statistics[i].operation == -1){
            continue;
        }

        char *operation = get_bbop_name(bbop_statistics[i].operation);
        
        { // SIMDRAM_1  
            bbop_statistics[i].simdram_1_subarray_latency = bbop_statistics[i].simdram_1_subarray_latency/1000000;
            bbop_statistics[i].simdram_1_subarray_energy = bbop_statistics[i].simdram_1_subarray_energy/1000000;
            
        
            // Write the statistics of the SIMDRAM_1
            fprintf(fp, "bbop_%d, %s, %ld, %ld, SIMDRAM_1, %f, %f\n", i, operation, bbop_statistics[i].largest_input_a, bbop_statistics[i].largest_input_b, bbop_statistics[i].simdram_1_subarray_latency, bbop_statistics[i].simdram_1_subarray_energy);

            // Update the summary statistics
            simdram_1_subarray_latency_total += bbop_statistics[i].simdram_1_subarray_latency;
            simdram_1_subarray_energy_total += bbop_statistics[i].simdram_1_subarray_energy;

        } 
        { // SIMDRAM_64  
            bbop_statistics[i].simdram_64_subarray_latency = bbop_statistics[i].simdram_64_subarray_latency/1000000;
            bbop_statistics[i].simdram_64_subarray_energy = bbop_statistics[i].simdram_64_subarray_energy/1000000;

            // Write the statistics of the SIMDram
            fprintf(fp, "bbop_%d, %s, %ld, %ld, SIMDRAM_64, %f, %f\n", i, operation, bbop_statistics[i].largest_input_a, bbop_statistics[i].largest_input_b, bbop_statistics[i].simdram_64_subarray_latency, bbop_statistics[i].simdram_64_subarray_energy);

            // Update the summary statistics
            simdram_64_subarray_latency_total += bbop_statistics[i].simdram_64_subarray_latency;
            simdram_64_subarray_energy_total += bbop_statistics[i].simdram_64_subarray_energy;
        } 
        { // SIMDRAM_64_DYNAMIC 
            bbop_statistics[i].simdram_64_subarray_dynamic_precision_latency = bbop_statistics[i].simdram_64_subarray_dynamic_precision_latency/1000000;
            bbop_statistics[i].simdram_64_subarray_dynamic_precision_energy = bbop_statistics[i].simdram_64_subarray_dynamic_precision_energy/1000000;

            // Write the statistics of the SIMDRAM_64_DYNAMIC
            fprintf(fp, "bbop_%d, %s, %ld, %ld, SIMDRAM_64_DYNAMIC, %f, %f\n", i, operation, bbop_statistics[i].largest_input_a, bbop_statistics[i].largest_input_b, bbop_statistics[i].simdram_64_subarray_dynamic_precision_latency, bbop_statistics[i].simdram_64_subarray_dynamic_precision_energy);

            // Update the summary statistics
            simdram_64_subarray_dynamic_precision_latency_total += bbop_statistics[i].simdram_64_subarray_dynamic_precision_latency;
            simdram_64_subarray_dynamic_precision_energy_total += bbop_statistics[i].simdram_64_subarray_dynamic_precision_energy;
        } 
        {  // DAFTPUM_STATIC_LAT
            bbop_statistics[i].daftpum_static_latency_optimized_latency = bbop_statistics[i].daftpum_static_latency_optimized_latency/1000000;
            bbop_statistics[i].daftpum_static_latency_optimized_energy = bbop_statistics[i].daftpum_static_latency_optimized_energy/1000000;

            // Write the statistics of the DAFTPUM_STATIC_LAT
            fprintf(fp, "bbop_%d, %s, %ld, %ld, DAFTPUM_STATIC_LAT, %f, %f\n", i, operation, bbop_statistics[i].largest_input_a, bbop_statistics[i].largest_input_b, bbop_statistics[i].daftpum_static_latency_optimized_latency, bbop_statistics[i].daftpum_static_latency_optimized_energy);

            // Update the summary statistics
            daftpum_static_latency_optimized_latency_total += bbop_statistics[i].daftpum_static_latency_optimized_latency;
            daftpum_static_latency_optimized_energy_total += bbop_statistics[i].daftpum_static_latency_optimized_energy;
        } 
        {  // DAFTPUM_LAT
            bbop_statistics[i].daftpum_latency_optimized_latency = bbop_statistics[i].daftpum_latency_optimized_latency/1000000;
            bbop_statistics[i].daftpum_latency_optimized_energy = bbop_statistics[i].daftpum_latency_optimized_energy/1000000;

            // Write the statistics of the DAFTPUM_LAT
            fprintf(fp, "bbop_%d, %s, %ld, %ld, DAFTPUM_LAT, %f, %f\n", i, operation, bbop_statistics[i].largest_input_a, bbop_statistics[i].largest_input_b, bbop_statistics[i].daftpum_latency_optimized_latency, bbop_statistics[i].daftpum_latency_optimized_energy);

            // Update the summary statistics
            daftpum_latency_optimized_latency_total += bbop_statistics[i].daftpum_latency_optimized_latency;
            daftpum_latency_optimized_energy_total += bbop_statistics[i].daftpum_latency_optimized_energy;
        } 
        {  // DAFTPUM_STATIC_ENE
            bbop_statistics[i].daftpum_static_energy_optimized_latency = bbop_statistics[i].daftpum_static_energy_optimized_latency/1000000;
            bbop_statistics[i].daftpum_static_energy_optimized_energy = bbop_statistics[i].daftpum_static_energy_optimized_energy/1000000;

            // Write the statistics of the DAFTPUM_STATIC_ENE
            fprintf(fp, "bbop_%d, %s, %ld, %ld, DAFTPUM_STATIC_ENE, %f, %f\n", i, operation, bbop_statistics[i].largest_input_a, bbop_statistics[i].largest_input_b, bbop_statistics[i].daftpum_static_energy_optimized_latency, bbop_statistics[i].daftpum_static_energy_optimized_energy);

            // Update the summary statistics
            daftpum_static_energy_optimized_latency_total += bbop_statistics[i].daftpum_static_energy_optimized_latency;
            daftpum_static_energy_optimized_energy_total += bbop_statistics[i].daftpum_static_energy_optimized_energy;
        } 
        {  // DAFTPUM_ENE
            bbop_statistics[i].daftpum_energy_optimized_latency = bbop_statistics[i].daftpum_energy_optimized_latency/1000000;
            bbop_statistics[i].daftpum_energy_optimized_energy = bbop_statistics[i].daftpum_energy_optimized_energy/1000000;

            // Write the statistics of the DAFTPUM_ENE
            fprintf(fp, "bbop_%d, %s, %ld, %ld, DAFTPUM_ENE, %f, %f\n", i, operation, bbop_statistics[i].largest_input_a, bbop_statistics[i].largest_input_b, bbop_statistics[i].daftpum_energy_optimized_latency, bbop_statistics[i].daftpum_energy_optimized_energy);

            // Update the summary statistics
            daftpum_energy_optimized_latency_total += bbop_statistics[i].daftpum_energy_optimized_latency;
            daftpum_energy_optimized_energy_total += bbop_statistics[i].daftpum_energy_optimized_energy;
        } 
        {  // DAFTPUM_TFAW
            bbop_statistics[i].daftpum_tfaw_enabled_latency = bbop_statistics[i].daftpum_tfaw_enabled_latency/1000000;
            bbop_statistics[i].daftpum_tfaw_enabled_energy = bbop_statistics[i].daftpum_tfaw_enabled_energy/1000000;

            // Write the statistics of the SIMDram
            fprintf(fp, "bbop_%d, %s, %ld, %ld, DAFTPUM_TFAW, %f, %f\n", i, operation, bbop_statistics[i].largest_input_a, bbop_statistics[i].largest_input_b, bbop_statistics[i].daftpum_tfaw_enabled_latency, bbop_statistics[i].daftpum_tfaw_enabled_energy);

            // Update the summary statistics
            daftpum_tfaw_enabled_latency_total += bbop_statistics[i].daftpum_tfaw_enabled_latency;
            daftpum_tfaw_enabled_energy_total += bbop_statistics[i].daftpum_tfaw_enabled_energy;
        } 

    }

    // Write the summary statistics
    fprintf(fp, "Summary, , , , SIMDRAM_1, %f, %f, \n", simdram_1_subarray_latency_total, simdram_1_subarray_energy_total);
    fprintf(fp, "Summary, , , , SIMDRAM_64, %f, %f, \n", simdram_64_subarray_latency_total, simdram_64_subarray_energy_total);
    fprintf(fp, "Summary, , , , SIMDRAM_64_DYNAMIC, %f, %f, \n", simdram_64_subarray_dynamic_precision_latency_total, simdram_64_subarray_dynamic_precision_energy_total);
    fprintf(fp, "Summary, , , , DAFTPUM_STATIC_LAT, %f, %f, \n", daftpum_static_latency_optimized_latency_total, daftpum_static_latency_optimized_energy_total);
    fprintf(fp, "Summary, , , , DAFTPUM_STATIC_ENE, %f, %f, \n", daftpum_static_energy_optimized_latency_total, daftpum_static_energy_optimized_energy_total);
    fprintf(fp, "Summary, , , , DAFTPUM_LAT, %f, %f, \n", daftpum_latency_optimized_latency_total, daftpum_latency_optimized_energy_total);
    fprintf(fp, "Summary, , , , DAFTPUM_ENE, %f, %f, \n", daftpum_energy_optimized_latency_total, daftpum_energy_optimized_energy_total);
    fprintf(fp, "Summary, , , , DAFTPUM_TFAW, %f, %f, \n", daftpum_tfaw_enabled_latency_total, daftpum_tfaw_enabled_energy_total);

    // Close the file
    fclose(fp);

    // close the resolution file
    #ifdef PROFILE_RESOLUTION
        // Write the resolution histogram to the file
        // header: resolution, count
        fprintf(resolution, "resolution, count\n");
        for (int i = 0; i < 65; i++){            
            fprintf(resolution, "%d, %ld\n", i, resolution_histogram[i]);
        }
    #endif 

    printf("[DEBUG] Done printing the statistics of the BBOPs\n");
    
}


#define FIXED_BIT 12

unsigned short int float2fix(float n)
{
    unsigned short int int_part = 0, frac_part = 0;
    int i;
    float t;

    int_part = (int)floor(n) << FIXED_BIT;
    n -= (int)floor(n);

    t = 0.5;
    for (i = 0; i < FIXED_BIT; i++) {
        if ((n - t) >= 0) {
            n -= t;
            frac_part += (1 << (FIXED_BIT - 1 - i));
        }
        t = t /2;
    }

    return int_part + frac_part;
}
