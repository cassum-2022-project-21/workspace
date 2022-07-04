#include <rebound.h>
#include <math.h>

#include "common.h"
#include "drag_force.h"
#include "profiles.h"

double DRAG_COEFF;

void IOPF_drag_force(struct reb_particle* const p, struct IOPF_drag_force_diag* diag) {
    double d, ux, uy;
    mag_dir_2d(p->x, p->y, &d, &ux, &uy);

    struct interp_loc iloc = interp_locate(d, STD_PROF_X, STD_PROF_N);

    // double vt_gas = interp_eval_cubic(iloc, VELOCITY_PROF[0]);
    // double vr_gas = interp_eval_cubic(iloc, VELOCITY_PROF[1]);
    // double rho_0 = interp_eval_cubic(iloc, DENSITY_PROF);

    double vt_gas = 0.99 * 2 * IOPF_PI / sqrt(d);
    double vr_gas = 0.0;
    double rho_0 =  8.41644500e-03;

    double vx_gas = ux * vr_gas - uy * vt_gas;
    double vy_gas = uy * vr_gas + ux * vt_gas;

    double vx_rel = vx_gas - p->vx;
    double vy_rel = vy_gas - p->vy;
    
    double v_rel, ux_rel, uy_rel;
    mag_dir_2d(vx_rel, vy_rel, &v_rel, &ux_rel, &uy_rel);

    double a_d = (0.5 * DRAG_COEFF * rho_0 * v_rel * v_rel * IOPF_PI * p->r * p->r) / p->m;

    if (diag) {
        struct reb_particle com = reb_get_com(p->sim);
        diag->orbit = reb_tools_particle_to_orbit(p->sim->G, *p, com);

        diag->vt_gas = vt_gas;
        diag->vr_gas = vr_gas;
        diag->rho_0 = rho_0;

        diag->vx_gas = vx_gas;
        diag->vy_gas = vy_gas;
        
        diag->v_gas = sqrt(vx_gas * vx_gas + vy_gas * vy_gas);
        diag->v_rel = v_rel;

        diag->ux_rel = ux_rel;
        diag->uy_rel = uy_rel;

        diag->a_d = a_d;
        
        diag->P_d = (ux_rel * p->vx + uy_rel * p->vy) * a_d * p->m;
    } else {
        p->ax += a_d * ux_rel;
        p->ay += a_d * uy_rel;
    }
}
