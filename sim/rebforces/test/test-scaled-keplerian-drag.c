#include <stdio.h>
#include <stdlib.h>
#include <rebound.h>
#include <math.h>

#include "../common.h"
#include "../forces.h"
#include "../profiles.h"
#include "../drag_force.h"

// double cAU = 14959787070000;
// double cYR = 365.2564 * 86400;
// double cMS = 1.9885e33;

double cAU = 1.0;
double cYR = 1.0;
double cMS = 1.0;

#define cV cAU / cYR
#define cE cMS * cAU * cAU / (cYR * cYR)
#define cD cMS / (cAU * cAU * cAU)

double AI;
double EI;
double TI = 0.0;
double diag_time = -1;

void heartbeat(struct reb_simulation *reb_sim) {
    if (reb_sim->t > diag_time) {
        struct IOPF_drag_force_diag diag;
        struct reb_particle *p = reb_sim->particles + 1;
        IOPF_drag_force(reb_sim->particles + 1, &diag);

        diag_time=reb_sim->t+(reb_sim->t < 2 ? 0.00625 : 2000.0);

        double E = reb_tools_energy(reb_sim);
        fprintf(stdout, "t=%f, ΔE/Δt=%.5e\n", reb_sim->t, (E-EI)/(reb_sim->t-TI) * cE);
        EI=E;
        TI=reb_sim->t;
        fprintf(stdout, "Planet: d=%.5f, x=%.5e, y=%.5e, vx=%.5e, vy=%.5e, v=%.5e.\n", diag.orbit.d, p->x * cAU, p->y * cAU, p->vx * cV, p->vy * cV, diag.orbit.v * cV);
        fprintf(stdout, "Interpolated: vt_gas=%.5e, vr_gas=%.5e, rho_0=%.5e\n", diag.vt_gas * cV, diag.vr_gas * cV, diag.rho_0 * cD);
        fprintf(stdout, "Gas: vx=%.5e, vy=%.5e, v=%.5e. Relative: v=%.5e, ux=%.5f, uy=%.5f\n", diag.vx_gas * cV, diag.vy_gas * cV, diag.v_gas * cV, diag.v_rel * cV, diag.ux_rel, diag.uy_rel);
        fprintf(stdout, "Drag acceleration: a=%.5e, dE/dt=%.5e\n", diag.a_d * cV / cYR, diag.P_d * cE / cYR);
        fprintf(stdout, "Orbit: a=%.9f, Δa=%.5e, e=%.5f, P=%.5f\n\n", diag.orbit.a, diag.orbit.a-AI, diag.orbit.e, diag.orbit.P);
        AI = diag.orbit.a;
    }
}

int main(int argc, char** argv) {
    double SCALE = atof(argv[1]);

    double* xp = STD_PROF_X; double* vtp = VELOCITY_PROF[0]; double* vrp = VELOCITY_PROF[1]; double* dp = DENSITY_PROF;
    uint_fast8_t n = 0;
    for (short i = 0; i < STD_PROF_NMAX; i++) {
        double x = 0.10 + (0.20 / (STD_PROF_NMAX-1)) * i;
        *xp++ = x; *vtp++ = (SCALE * 2 * IOPF_PI) / sqrt(x); *vrp++ = 0.0; *dp++ = 5.0e-9 * G_PER_CM3; n++;
    }
    STD_PROF_N = n;
    DRAG_COEFF = 1.0;

    struct reb_simulation *reb_sim = reb_create_simulation();

    reb_sim->integrator = REB_INTEGRATOR_IAS15;
    reb_sim->G = 4. * IOPF_PI * IOPF_PI;
    // reb_sim->collision = REB_COLLISION_DIRECT;
    // reb_sim->collision_resolve = reb_collision_resolve_merge;

    reb_add_fmt(reb_sim, "m", 1.);               // Central object of 1 solar mass
    reb_add_fmt(reb_sim, "m a e r", 3e-6, 0.25, 0.0, 4.26352e-5); // Planet orbiting at 1 AU, ~1 earth mass, 0 eccentricity

    reb_sim->heartbeat = heartbeat;
    reb_sim->additional_forces = IOPF_drag_all;
    reb_sim->force_is_velocity_dependent = 1;

    reb_move_to_com(reb_sim);

    EI=reb_tools_energy(reb_sim);
    struct reb_particle com = reb_get_com(reb_sim);
    struct reb_orbit orbit = reb_tools_particle_to_orbit(reb_sim->G, reb_sim->particles[1], com);
    AI=orbit.a;

    reb_integrate(reb_sim, 2);

    reb_free_simulation(reb_sim);
}
