#ifndef IOPF_PROFILES
#define IOPF_PROFILES

#include <stddef.h>

extern double VELOCITY_PROF[2][64];
extern size_t VELOCITY_PROF_N;

extern double DENSITY_PROF[64];
extern size_t DENSITY_PROF_N;

struct interp_loc {
    size_t idx;
    double s;
};

// locate_* require x to be sorted increasing
struct interp_loc interp_locate_linear(double x, double* xs, size_t N);
struct interp_loc interp_locate_binary(double x, double* xs, size_t N);

double evaluate(struct interp_loc loc, double* xs);

#ifdef IOPF_BINARY_LOCATE
#define interp_locate interp_locate_binary
#else
#define interp_locate interp_locate_linear
#endif

#endif
