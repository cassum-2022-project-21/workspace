#include <stdio.h>
#include <stdlib.h>
#include <rebound.h>
#include <math.h>

#include "../common.h"

#define SCALE 1.01 /* Super-Keplerian gas */
// #define SCALE 0.99 /* Sub-Keplerian gas */

void heartbeat(struct reb_simulation *reb_sim) {
    if (reb_output_check(reb_sim, 1000. * 0.618033989)) {
        struct reb_particle* p = &reb_sim->particles[1];
        double r, ux, uy;
        mag_dir_2d(p->x, p->y, &r, &ux, &uy);

        double v_k = (SCALE * 2 * IOPF_PI) / sqrt(r);
        double vx_k = -v_k * uy;
        double vy_k = v_k * ux;

        double vx_rel = vx_k - p->vx;
        double vy_rel = vy_k - p->vy;
        
        double v_rel2, ux_rel, uy_rel;
        mag2_dir_2d(vx_rel, vy_rel, &v_rel2, &ux_rel, &uy_rel);

        double a_d = (0.5 * 1e-8 * G_PER_CM3 * v_rel2 * IOPF_PI * p->r * p->r) / p->m;

        fprintf(stdout, "t=%f. Planet: r=%.5f, x=%.5f, y=%.5f, vx=%.5f, vy=%.5f.\n", reb_sim->t, r, p->x, p->y, p->vx, p->vy);
        fprintf(stdout, "Sub-Keplerian: vx=%.5f, vy=%.5f. Relative: vx=%.5e, vy=%.5e\n", vx_k, vy_k, vx_rel, vy_rel);
        fprintf(stdout, "Drag acceleration: ax=%.5e, ay=%.5e\n", a_d * ux_rel, a_d * uy_rel);
        
        struct reb_orbit orbit = reb_tools_particle_to_orbit(reb_sim->G, reb_sim->particles[0], reb_sim->particles[1]);
        fprintf(stdout, "Orbit: a=%.5f, e=%.5f\n", orbit.a, orbit.e);
    }
}

void sub_keplerian_drag(struct reb_simulation *reb_sim) {
    struct reb_particle* p = &reb_sim->particles[1];
    double r, ux, uy;
    mag_dir_2d(p->x, p->y, &r, &ux, &uy);

    double v_k = (SCALE * 2 * IOPF_PI) / sqrt(r);
    double vx_k = -v_k * uy;
    double vy_k = v_k * ux;

    double vx_rel = vx_k - p->vx;
    double vy_rel = vy_k - p->vy;
    
    double v_rel2, ux_rel, uy_rel;
    mag2_dir_2d(vx_rel, vy_rel, &v_rel2, &ux_rel, &uy_rel);

    double a_d = (0.5 * 1e-8 * G_PER_CM3 * v_rel2 * IOPF_PI * p->r * p->r) / p->m;

    p->ax += a_d * ux_rel;
    p->ay += a_d * uy_rel;
}

int main(int argc, char **argv) {
    struct reb_simulation *reb_sim = reb_create_simulation();

    reb_sim->integrator = REB_INTEGRATOR_IAS15;
    reb_sim->G = 4. * IOPF_PI * IOPF_PI;
    reb_sim->collision = REB_COLLISION_DIRECT;
    reb_sim->collision_resolve = reb_collision_resolve_merge;

    reb_add_fmt(reb_sim, "m", 1.);               // Central object of 1 solar mass
    reb_add_fmt(reb_sim, "m a e r", 3e-6, 0.1, 0.2, 4.26352e-5); // Planet orbiting at 1 AU, ~1 earth mass, 0 eccentricity

    reb_sim->heartbeat = heartbeat;
    reb_sim->additional_forces = sub_keplerian_drag;
    reb_sim->force_is_velocity_dependent = 1;

    reb_move_to_com(reb_sim);

    reb_integrate(reb_sim, INFINITY);

    reb_free_simulation(reb_sim);
}
