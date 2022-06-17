#include <stdio.h>
#include <stdlib.h>
#include <rebound.h>
#include <math.h>

#include "../common.h"

void heartbeat(struct reb_simulation *reb_sim) {
    if (reb_output_check(reb_sim, 1000. * 0.618033989)) {
        struct reb_particle s = reb_sim->particles[0];
        struct reb_particle p = reb_sim->particles[1];
        double r, ux, uy;
        mag_dir_2d(p.x, p.y, &r, &ux, &uy);

        double v_k =  sqrt(reb_sim->G * (p.m + s.m) / r);
        double vx_k = -v_k * uy;
        double vy_k = v_k * ux;

        fprintf(stdout, "t=%f. Planet: x=%.8f, y=%.8f, vx=%.8f, vy=%.8f. ", reb_sim->t, p.x, p.y, p.vx, p.vy);
        fprintf(stdout, "Keplerian: vx=%.8f, vy=%.8f. Relative: vx=%.5e, vy=%.5e\n", vx_k, vy_k, vx_k-p.vx, vy_k-p.vy);
    }
}

int main(int argc, char **argv) {
    struct reb_simulation *reb_sim = reb_create_simulation();

    reb_sim->integrator = REB_INTEGRATOR_IAS15;
    reb_sim->G = 4. * IOPF_PI * IOPF_PI;
    reb_sim->collision = REB_COLLISION_DIRECT;
    reb_sim->collision_resolve = reb_collision_resolve_merge;

    reb_add_fmt(reb_sim, "m", 1.);               // Central object of 1 solar mass
    reb_add_fmt(reb_sim, "m a e r", 3e-6, 0.1, 0., 4.26352e-5); // Planet orbiting at 1 AU, ~1 earth mass, 0 eccentricity

    // reb_sim->heartbeat = heartbeat;

    reb_move_to_com(reb_sim);

    reb_integrate(reb_sim, 1000);

    reb_free_simulation(reb_sim);
}
