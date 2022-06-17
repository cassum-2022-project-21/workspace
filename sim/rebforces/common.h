#ifndef IOPF_COMMON
#define IOPF_COMMON

#include <inttypes.h>

struct interp_loc {
    double s;
    uint_fast8_t idx;
};

// locate_* require x to be sorted increasing
struct interp_loc interp_locate_linear(const double x, const double* const xs, const uint_fast8_t N);
struct interp_loc interp_locate_binary(const double x, const double* const xs, const uint_fast8_t N);

#define interp_locate interp_locate_binary

double interp_eval(const struct interp_loc loc, const double* const ys);

inline void mag_dir_2d(const double x, const double y, double* const r, double* const ux, double* const uy) {
    *r = sqrt(x*x + y*y);
    *ux = x / *r;
    *uy = y / *r;
}

inline void mag2_dir_2d(const double x, const double y, double* const r2, double* const ux, double* const uy) {
    *r2 = x*x + y*y;
    double isr = 1.0 / sqrt(*r2);
    *ux = x * isr;
    *uy = y * isr;
}

extern const double IOPF_PI; /* M_PI on GNU. consistency if we run on other system */
extern const double CM_PER_S; /* 1 cm/s to AU/sidereal year */
extern const double G_PER_CM3; /* 1 g / cm^3 to solar mass / AU^3 */

#endif
