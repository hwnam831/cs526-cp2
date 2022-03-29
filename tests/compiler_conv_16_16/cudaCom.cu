
#include <stdio.h>
#include <time.h>

#include <cutil.h>

#include "cudaCom.h"

// current timer
unsigned int timer = 0;

/**
 * print the interval time between two time call of this function.
 * When we first call this function, it will create timer. 
 * The last call should set msg be null to destroy the timer.
 *! @param msg  : the msg of the interval time
 */
float timestamp(char* msg) {

	float f = 0;
	if (timer!=0) {	// destroy old timer and create new timer
	    // stop and destroy timer		
	    CUT_SAFE_CALL(cutStopTimer(timer));

		//output the time
		f = cutGetTimerValue(timer);
//	    printf("Processing time [%s]: %f (ms)\n", msg?msg:"end", f);
	    CUT_SAFE_CALL(cutDeleteTimer(timer));
	}
	else { // first time  
//	    printf("Start Timer\n");		
	}
	
	// start timer
	if (msg) {
		CUT_SAFE_CALL(cutCreateTimer(&timer));
		CUT_SAFE_CALL(cutStartTimer(timer));
	}
	return f;

}

