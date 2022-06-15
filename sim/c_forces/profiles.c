#include <math.h>

#include "profiles.h"

#define STD_PROF_X_N 64
double STD_PROF_X[STD_PROF_X_N];
double VELOCITY_PROF[2][STD_PROF_X_N];
double DENSITY_PROF[STD_PROF_X_N];

struct interp_loc interp_locate_linear(double x, double* xs, size_t N) {
    size_t i;
    for (i = 1; i < N; i++) {
        if (x < xs[i]) {
            return (struct interp_loc){ i-1, (x - xs[i-1])/(xs[i] - xs[i-1]) };
        }
    }
    return (struct interp_loc){ N , NAN };
}

double interp_eval(struct interp_loc loc, double* ys) {
    return ys[loc.idx] + loc.s * (ys[loc.idx+1] - ys[loc.idx]);
}
