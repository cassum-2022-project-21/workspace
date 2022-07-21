#include <math.h>   
#include <stddef.h>

#include "profiles.h"

unsigned int STD_PROF_N = 0;

double STD_PROF_X[STD_PROF_NMAX] = { NAN }; // units: AU
double VELOCITY_PROF[2][STD_PROF_NMAX] = { { NAN } }; // units: AU / yr
double DENSITY_PROF[STD_PROF_NMAX] = { NAN }; // units: Msun / AU^3

double SURFACE_DENSITY_PROF[STD_PROF_NMAX] = { NAN }; // units: Msun / AU^2
double SCALE_HEIGHT_PROF[STD_PROF_NMAX] = { NAN }; // dimensionless
double TORQUE_PROF[STD_PROF_NMAX] = { NAN }; // units: Gamma_0 (standard scaling torque)
