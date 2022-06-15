#include <inttypes.h>
#include <math.h>

#include "common.h"

struct interp_loc interp_locate_linear(const double x, const double* const xs, const uint_fast8_t N) {
    uint_fast8_t i;
    for (i = 1; i < N; i++) {
        if (x < xs[i]) {
            return (struct interp_loc){ (x - xs[i-1])/(xs[i] - xs[i-1]), i-1 };
        }
    }
    return (struct interp_loc){ NAN, N };
}

double interp_eval(const struct interp_loc loc, const double* const ys) {
    return ys[loc.idx] + loc.s * (ys[loc.idx+1] - ys[loc.idx]);
}

extern inline void mag_dir_2d(const double x, const double y, double* const r, double* const ux, double* const uy);
extern inline void mag2_dir_2d(const double x, const double y, double* const r2, double* const ux, double* const uy);
