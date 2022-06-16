#include <math.h>   
#include <stddef.h>

#include "profiles.h"

uint_fast8_t STD_PROF_N = 0;

double STD_PROF_X[STD_PROF_NMAX] = { NAN };
double VELOCITY_PROF[2][STD_PROF_NMAX] = { { NAN } };
double DENSITY_PROF[STD_PROF_NMAX] = { NAN };
