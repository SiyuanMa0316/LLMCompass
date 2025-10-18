
#ifndef _TIMER_H_
#define _TIMER_H_ 1

#include <stdio.h>
#include <sys/time.h>

typedef struct Timer {
    struct timeval startTime;
    struct timeval endTime;
} Timer;


#ifdef GET_TIME_FUNCTION
    /*----------------------------------------------
    Setup variable declaration macros.
    ----------------------------------------------*/
    #ifndef VAR_DECLS
        # define _DECL extern
        # define _INIT(x)
    #else
        # define _DECL
        # define _INIT(x)  = x
    #endif

    // create a float variable to store the elapsed time in milliseconds
    _DECL float var_a _INIT(0.0);

    // create a function to increment the elapsed time by delta
    static void incrementElapsedTime(Timer timer) {
        float delta = (float) ((timer.endTime.tv_sec - timer.startTime.tv_sec)
                    + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6);
        var_a += delta;

    }

    // create a function to print the elapsed time
    static void printElapsedTimeCounter() {
        printf("Elapsed time: %f ms\n", var_a*1e3);
    }
#endif 

static void startTimer(Timer* timer) {
    gettimeofday(&(timer->startTime), NULL);
}

static void stopTimer(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
}

static float getElapsedTime(Timer timer) {
    return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec)
                   + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
}

static void printElapsedTime(Timer timer) {
    printf("Elapsed time: %f ms\n", getElapsedTime(timer)*1e3);
}


#endif
