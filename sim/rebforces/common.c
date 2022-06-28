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

struct interp_loc interp_locate_binary(const double x, const double* const xs, const uint_fast8_t N) {
    uint_fast8_t begin = 0;
    uint_fast8_t end = N;

    while (begin < end) {
        uint_fast8_t mid = (begin + end) / 2;
        if (xs[mid] >= x) {
            end = mid;
        } else {
            begin = mid + 1;
        }
    }

    return (struct interp_loc){ (x - xs[end-1])/(xs[end] - xs[end-1]), end-1 };
}

double interp_eval(const struct interp_loc loc, const double* const ys) {
    return ys[loc.idx] + loc.s * (ys[loc.idx+1] - ys[loc.idx]);
}

double interp_eval_cubic(const struct interp_loc loc, const double* const ys) {
    double s2 = loc.s * loc.s;

    double y0 = ys[loc.idx-1];
    double y1 = ys[loc.idx];
    double y2 = ys[loc.idx+1];
    double y3 = ys[loc.idx+2];

    double a0 = y3 - y2 - y0 + y1;
    double a1 = y0 - y1 - a0;
    double a2 = y2 - y0;

    return(
        a0 * loc.s * s2
        + a1 * s2
        + a2 * loc.s
        + y1);
}

extern inline void mag_dir_2d(const double x, const double y, double* const r, double* const ux, double* const uy);
extern inline void mag2_dir_2d(const double x, const double y, double* const r2, double* const ux, double* const uy);

const double IOPF_PI = 3.14159265358979323846;
const double CM_PER_S = 2.109532e-6;
const double G_PER_CM3 = 1.683289e+6;
