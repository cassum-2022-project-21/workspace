#ifndef IOPF_DRAG_FORCE
#define IOPF_DRAG_FORCE

#include <rebound.h>

extern double DRAG_COEFF;

struct IOPF_drag_force_diag {
    struct reb_orbit orbit; // Orbital elements of the planet

    double vt_gas; // Tangential velocity of gas at d
    double vr_gas; // Radial velocity of gas at d
    double rho_0; // Density

    double vx_gas; // x velocity of gas
    double vy_gas; // y velocity of gas

    double v_gas; // Velocity of the gas
    double v_rel; // Relative velocity of planet and gas

    double ux_rel; // x component of unit normal relative velocity
    double uy_rel; // y component of unit normal relative velocity

    double a_d; // Acceleration of the drag force
    double P_d; // Power of the drag force
};

void IOPF_drag_force(struct reb_particle* const p, struct IOPF_drag_force_diag* diag);

#endif
