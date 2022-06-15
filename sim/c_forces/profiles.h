#ifndef IOPF_PROFILES
#define IOPF_PROFILES

#include <inttypes.h>

#define STD_PROF_NMAX __UINT_FAST8_MAX__ /* The maximum profile length we accept */

extern uint_fast8_t STD_PROF_X_N; /* The actual length of the profiles loaded (user-defined) */

extern double STD_PROF_X[STD_PROF_NMAX]; /* The radii of the velocity/density/etc. samples */
extern double VELOCITY_PROF[2][STD_PROF_NMAX]; /* The gas velocity profile at the radii in STD_PROF_X. VELOCITY_PROF[0] is tangential, VELOCITY_PROF[1] is radial */
extern double DENSITY_PROF[STD_PROF_NMAX]; /* The gas density profile at the radii in STD_PROF_X */

#endif
