#include <rebound.h>
#include <math.h>

#include "common.h"
#include "drag_force.h"
#include "profiles.h"

double DRAG_COEFF;

void IOPF_drag_force(struct reb_particle* const p) {
    double r, ux, uy;
    mag_dir_2d(p->x, p->y, &r, &ux, &uy);

    struct interp_loc iloc = interp_locate(r, STD_PROF_X, STD_PROF_N);

    double vt_gas = interp_eval(iloc, VELOCITY_PROF[0]);
    double vr_gas = interp_eval(iloc, VELOCITY_PROF[1]);
    double rho_0 = interp_eval(iloc, DENSITY_PROF);

    double vx_gas = ux * vr_gas - uy * vt_gas;
    double vy_gas = uy * vr_gas + ux * vt_gas;

    double vx_rel = vx_gas - p->vx;
    double vy_rel = vy_gas - p->vy;
    
    double v_rel2, ux_rel, uy_rel;
    mag2_dir_2d(vx_rel, vy_rel, &v_rel2, &ux_rel, &uy_rel);

    double a_d = (0.5 * rho_0 * v_rel2 * IOPF_PI * p->r * p->r) / p->m;

    p->ax += a_d * ux_rel;
    p->ay += a_d * uy_rel;
}
