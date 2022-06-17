#include <stdio.h>
#include <stdlib.h>
#include <rebound.h>

#include "../forces.h"
#include "../profiles.h"
#include "../common.h"

void heartbeat(struct reb_simulation *reb_sim) {
    if (reb_output_check(reb_sim, 1000. * 0.618033989)) {
        struct reb_particle* p = &reb_sim->particles[1];
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

        double a_d = (0.5 * 1e-8 * G_PER_CM3 * v_rel2 * IOPF_PI * p->r * p->r) / p->m;

        fprintf(stdout, "t=%f. Planet: r=%.5f, x=%.5f, y=%.5f, vx=%.5f, vy=%.5f.\n", reb_sim->t, r, p->x, p->y, p->vx, p->vy);
        fprintf(stdout, "Interpolated: vx=%.5f, vy=%.5f. Relative: vx=%.5e, vy=%.5e\n", vx_gas, vy_gas, vx_rel, vy_rel);
        fprintf(stdout, "Drag acceleration: ax=%.5e, ay=%.5e\n", a_d * ux_rel, a_d * uy_rel);
        
        struct reb_orbit orbit = reb_tools_particle_to_orbit(reb_sim->G, reb_sim->particles[0], reb_sim->particles[1]);
        fprintf(stdout, "Orbit: a=%.5f, e=%.5f\n", orbit.a, orbit.e);
    }
}

int main(int argc, char** argv) {
    double* xp = STD_PROF_X; double* vtp = VELOCITY_PROF[0]; double* vrp = VELOCITY_PROF[1]; double* dp = DENSITY_PROF;
    uint_fast8_t n = 0;
    *xp++ = 9.903757e-02; *vtp++ = 9.471901e+06 * CM_PER_S; *vrp++ = 3.221833e+01 * CM_PER_S; *dp++ = 2.772388e-09 * G_PER_CM3; n++;
    *xp++ = 1.019076e-01; *vtp++ = 9.337649e+06 * CM_PER_S; *vrp++ = 2.916174e+01 * CM_PER_S; *dp++ = 2.854533e-09 * G_PER_CM3; n++;
    *xp++ = 1.048608e-01; *vtp++ = 9.205303e+06 * CM_PER_S; *vrp++ = 2.639224e+01 * CM_PER_S; *dp++ = 2.939481e-09 * G_PER_CM3; n++;
    *xp++ = 1.078996e-01; *vtp++ = 9.074835e+06 * CM_PER_S; *vrp++ = 2.388306e+01 * CM_PER_S; *dp++ = 3.027341e-09 * G_PER_CM3; n++;
    *xp++ = 1.110264e-01; *vtp++ = 8.946217e+06 * CM_PER_S; *vrp++ = 2.160990e+01 * CM_PER_S; *dp++ = 3.118236e-09 * G_PER_CM3; n++;
    *xp++ = 1.142439e-01; *vtp++ = 8.819426e+06 * CM_PER_S; *vrp++ = 1.955160e+01 * CM_PER_S; *dp++ = 3.212074e-09 * G_PER_CM3; n++;
    *xp++ = 1.175546e-01; *vtp++ = 8.694436e+06 * CM_PER_S; *vrp++ = 1.768740e+01 * CM_PER_S; *dp++ = 3.309117e-09 * G_PER_CM3; n++;
    *xp++ = 1.209612e-01; *vtp++ = 8.571220e+06 * CM_PER_S; *vrp++ = 1.599894e+01 * CM_PER_S; *dp++ = 3.409554e-09 * G_PER_CM3; n++;
    *xp++ = 1.244666e-01; *vtp++ = 8.449754e+06 * CM_PER_S; *vrp++ = 1.446981e+01 * CM_PER_S; *dp++ = 3.513519e-09 * G_PER_CM3; n++;
    *xp++ = 1.280735e-01; *vtp++ = 8.330011e+06 * CM_PER_S; *vrp++ = 1.308511e+01 * CM_PER_S; *dp++ = 3.621165e-09 * G_PER_CM3; n++;
    *xp++ = 1.317850e-01; *vtp++ = 8.211969e+06 * CM_PER_S; *vrp++ = 1.183133e+01 * CM_PER_S; *dp++ = 3.732633e-09 * G_PER_CM3; n++;
    *xp++ = 1.356040e-01; *vtp++ = 8.095603e+06 * CM_PER_S; *vrp++ = 1.069623e+01 * CM_PER_S; *dp++ = 3.848090e-09 * G_PER_CM3; n++;
    *xp++ = 1.395337e-01; *vtp++ = 7.980889e+06 * CM_PER_S; *vrp++ = 9.668670e+00 * CM_PER_S; *dp++ = 3.967695e-09 * G_PER_CM3; n++;
    *xp++ = 1.435773e-01; *vtp++ = 7.867802e+06 * CM_PER_S; *vrp++ = 8.738585e+00 * CM_PER_S; *dp++ = 4.091621e-09 * G_PER_CM3; n++;
    *xp++ = 1.477381e-01; *vtp++ = 7.756306e+06 * CM_PER_S; *vrp++ = 7.897191e+00 * CM_PER_S; *dp++ = 4.219760e-09 * G_PER_CM3; n++;
    *xp++ = 1.520194e-01; *vtp++ = 7.646425e+06 * CM_PER_S; *vrp++ = 7.135946e+00 * CM_PER_S; *dp++ = 4.352404e-09 * G_PER_CM3; n++;
    *xp++ = 1.564248e-01; *vtp++ = 7.538089e+06 * CM_PER_S; *vrp++ = 6.447131e+00 * CM_PER_S; *dp++ = 4.489889e-09 * G_PER_CM3; n++;
    *xp++ = 1.609579e-01; *vtp++ = 7.431292e+06 * CM_PER_S; *vrp++ = 5.823934e+00 * CM_PER_S; *dp++ = 4.632420e-09 * G_PER_CM3; n++;
    *xp++ = 1.656223e-01; *vtp++ = 7.326012e+06 * CM_PER_S; *vrp++ = 5.260177e+00 * CM_PER_S; *dp++ = 4.780216e-09 * G_PER_CM3; n++;
    *xp++ = 1.704219e-01; *vtp++ = 7.222226e+06 * CM_PER_S; *vrp++ = 4.750260e+00 * CM_PER_S; *dp++ = 4.933490e-09 * G_PER_CM3; n++;
    *xp++ = 1.753606e-01; *vtp++ = 7.119915e+06 * CM_PER_S; *vrp++ = 4.289104e+00 * CM_PER_S; *dp++ = 5.092478e-09 * G_PER_CM3; n++;
    *xp++ = 1.804424e-01; *vtp++ = 7.019058e+06 * CM_PER_S; *vrp++ = 3.872105e+00 * CM_PER_S; *dp++ = 5.257426e-09 * G_PER_CM3; n++;
    *xp++ = 1.856715e-01; *vtp++ = 6.919651e+06 * CM_PER_S; *vrp++ = 3.495087e+00 * CM_PER_S; *dp++ = 5.428584e-09 * G_PER_CM3; n++;
    *xp++ = 1.910521e-01; *vtp++ = 6.821656e+06 * CM_PER_S; *vrp++ = 3.153323e+00 * CM_PER_S; *dp++ = 5.608739e-09 * G_PER_CM3; n++;
    *xp++ = 1.965887e-01; *vtp++ = 6.725035e+06 * CM_PER_S; *vrp++ = 2.843694e+00 * CM_PER_S; *dp++ = 5.798322e-09 * G_PER_CM3; n++;
    *xp++ = 2.022857e-01; *vtp++ = 6.629787e+06 * CM_PER_S; *vrp++ = 2.563988e+00 * CM_PER_S; *dp++ = 5.995490e-09 * G_PER_CM3; n++;
    *xp++ = 2.081478e-01; *vtp++ = 6.535891e+06 * CM_PER_S; *vrp++ = 2.311354e+00 * CM_PER_S; *dp++ = 6.200602e-09 * G_PER_CM3; n++;
    *xp++ = 2.141797e-01; *vtp++ = 6.443330e+06 * CM_PER_S; *vrp++ = 2.083212e+00 * CM_PER_S; *dp++ = 6.414030e-09 * G_PER_CM3; n++;
    *xp++ = 2.203865e-01; *vtp++ = 6.352084e+06 * CM_PER_S; *vrp++ = 1.877222e+00 * CM_PER_S; *dp++ = 6.636168e-09 * G_PER_CM3; n++;
    *xp++ = 2.267731e-01; *vtp++ = 6.258068e+06 * CM_PER_S; *vrp++ = 1.691265e+00 * CM_PER_S; *dp++ = 6.867431e-09 * G_PER_CM3; n++;
    *xp++ = 2.333448e-01; *vtp++ = 6.159572e+06 * CM_PER_S; *vrp++ = 1.564110e+00 * CM_PER_S; *dp++ = 6.946203e-09 * G_PER_CM3; n++;
    *xp++ = 2.401070e-01; *vtp++ = 6.072233e+06 * CM_PER_S; *vrp++ = 1.540226e+00 * CM_PER_S; *dp++ = 6.650625e-09 * G_PER_CM3; n++;
    *xp++ = 2.470651e-01; *vtp++ = 5.986177e+06 * CM_PER_S; *vrp++ = 1.515473e+00 * CM_PER_S; *dp++ = 6.375671e-09 * G_PER_CM3; n++;
    *xp++ = 2.542248e-01; *vtp++ = 5.901272e+06 * CM_PER_S; *vrp++ = 1.489131e+00 * CM_PER_S; *dp++ = 6.124583e-09 * G_PER_CM3; n++;
    *xp++ = 2.615921e-01; *vtp++ = 5.817574e+06 * CM_PER_S; *vrp++ = 1.463078e+00 * CM_PER_S; *dp++ = 5.884634e-09 * G_PER_CM3; n++;
    *xp++ = 2.691728e-01; *vtp++ = 5.735066e+06 * CM_PER_S; *vrp++ = 1.437298e+00 * CM_PER_S; *dp++ = 5.655392e-09 * G_PER_CM3; n++;
    *xp++ = 2.769732e-01; *vtp++ = 5.653731e+06 * CM_PER_S; *vrp++ = 1.411774e+00 * CM_PER_S; *dp++ = 5.436437e-09 * G_PER_CM3; n++;
    *xp++ = 2.849997e-01; *vtp++ = 5.573553e+06 * CM_PER_S; *vrp++ = 1.386489e+00 * CM_PER_S; *dp++ = 5.227369e-09 * G_PER_CM3; n++;
    *xp++ = 2.932588e-01; *vtp++ = 5.494515e+06 * CM_PER_S; *vrp++ = 1.361426e+00 * CM_PER_S; *dp++ = 5.027810e-09 * G_PER_CM3; n++;
    *xp++ = 3.017572e-01; *vtp++ = 5.416593e+06 * CM_PER_S; *vrp++ = 1.336564e+00 * CM_PER_S; *dp++ = 4.837408e-09 * G_PER_CM3; n++;
    *xp++ = 3.105019e-01; *vtp++ = 5.339765e+06 * CM_PER_S; *vrp++ = 1.311886e+00 * CM_PER_S; *dp++ = 4.655822e-09 * G_PER_CM3; n++;
    *xp++ = 3.195000e-01; *vtp++ = 5.263962e+06 * CM_PER_S; *vrp++ = 1.288013e+00 * CM_PER_S; *dp++ = 4.479390e-09 * G_PER_CM3; n++;
    *xp++ = 3.287588e-01; *vtp++ = 5.189304e+06 * CM_PER_S; *vrp++ = 1.266118e+00 * CM_PER_S; *dp++ = 4.301905e-09 * G_PER_CM3; n++;
    *xp++ = 3.382860e-01; *vtp++ = 5.115708e+06 * CM_PER_S; *vrp++ = 1.244441e+00 * CM_PER_S; *dp++ = 4.132365e-09 * G_PER_CM3; n++;
    *xp++ = 3.480893e-01; *vtp++ = 5.043157e+06 * CM_PER_S; *vrp++ = 1.222967e+00 * CM_PER_S; *dp++ = 3.970448e-09 * G_PER_CM3; n++;
    *xp++ = 3.581766e-01; *vtp++ = 4.971638e+06 * CM_PER_S; *vrp++ = 1.201684e+00 * CM_PER_S; *dp++ = 3.815855e-09 * G_PER_CM3; n++;
    *xp++ = 3.685563e-01; *vtp++ = 4.901130e+06 * CM_PER_S; *vrp++ = 1.180578e+00 * CM_PER_S; *dp++ = 3.668299e-09 * G_PER_CM3; n++;
    *xp++ = 3.792368e-01; *vtp++ = 4.831637e+06 * CM_PER_S; *vrp++ = 1.159634e+00 * CM_PER_S; *dp++ = 3.527505e-09 * G_PER_CM3; n++;
    *xp++ = 3.902268e-01; *vtp++ = 4.763126e+06 * CM_PER_S; *vrp++ = 1.138838e+00 * CM_PER_S; *dp++ = 3.393216e-09 * G_PER_CM3; n++;
    *xp++ = 4.015352e-01; *vtp++ = 4.695575e+06 * CM_PER_S; *vrp++ = 1.118174e+00 * CM_PER_S; *dp++ = 3.265191e-09 * G_PER_CM3; n++;
    *xp++ = 4.131714e-01; *vtp++ = 4.628908e+06 * CM_PER_S; *vrp++ = 1.097910e+00 * CM_PER_S; *dp++ = 3.141980e-09 * G_PER_CM3; n++;
    STD_PROF_N = n;
    DRAG_COEFF = 1.0;

    struct reb_simulation *reb_sim = reb_create_simulation();

    reb_sim->integrator = REB_INTEGRATOR_IAS15;
    reb_sim->G = 4. * IOPF_PI * IOPF_PI;
    reb_sim->collision = REB_COLLISION_DIRECT;
    reb_sim->collision_resolve = reb_collision_resolve_merge;

    reb_add_fmt(reb_sim, "m", 1.);               // Central object of 1 solar mass
    reb_add_fmt(reb_sim, "m a e r", 3e-6, 0.3, 0.1, 4.26352e-5); // Planet orbiting at 1 AU, ~1 earth mass, 0 eccentricity

    reb_sim->heartbeat = heartbeat;
    reb_sim->additional_forces = IOPF_drag_all;
    reb_sim->force_is_velocity_dependent = 1;

    reb_move_to_com(reb_sim);

    reb_integrate(reb_sim, INFINITY);

    reb_free_simulation(reb_sim);
}
