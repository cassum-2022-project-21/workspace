#ifndef IOPF_DRAG_FORCE
#define IOPF_DRAG_FORCE

#include <rebound.h>

extern double DRAG_COEFF;

double IOPF_drag_force(struct reb_particle* const p);

#endif
