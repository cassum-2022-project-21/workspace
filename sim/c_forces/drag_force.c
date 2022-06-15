#include <rebound.h>
#include <math.h>

#include "drag_force.h"
#include "profiles.c"

void IOPF_drag_force(struct reb_particle* const p) {
    double r = sqrt(p->x * p->x + p->y * p->y);
    struct interp_loc iloc = interp_locate(r, ); 
}
