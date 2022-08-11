#include <rebound.h>

#include "common.h"
#include "profiles.h"
#include "migration_torque.h"

void IOPF_torque_all(struct reb_simulation* reb_sim) {
    struct reb_particle* const particles = reb_sim->particles;
    const int N = reb_sim->N;

    for (int i = 1; i < N; i++) {
        IOPF_torque_force(particles + i, particles);
    }
}

void IOPF_torque_force(struct reb_particle* const p, struct reb_particle* primary) {
    double mu,dx,dy,dz,dvx,dvy,dvz,d,vsquared,vcircsquared,a,hx,hy,hz,h;

    mu = p->sim->G*(p->m+primary->m);
    dx = p->x - primary->x;
    dy = p->y - primary->y;
    dz = p->z - primary->z;
    dvx = p->vx - primary->vx;
    dvy = p->vy - primary->vy;
    dvz = p->vz - primary->vz;
    d = sqrt (dx*dx + dy*dy + dz*dz);

    vsquared = dvx*dvx + dvy*dvy + dvz*dvz;
    vcircsquared = mu/d;	
    a = -mu/(vsquared - 2.*vcircsquared);

    hx = (dy*dvz - dz*dvy);
    hy = (dz*dvx - dx*dvz);
    hz = (dx*dvy - dy*dvx);
    h = sqrt(hx*hx + hy*hy + hz*hz);

    struct interp_loc iloc = interp_locate(a, STD_PROF_X, STD_PROF_N);

    double Omegasquared, hR, Gamma_r, Sigma, torque, t_m;
    hR = interp_eval(iloc, SCALE_HEIGHT_PROF);
    Sigma = interp_eval(iloc, SURFACE_DENSITY_PROF);
    torque = interp_eval(iloc, TORQUE_PROF);

    Omegasquared = mu / (a*a*a);
    Gamma_r = p->m * a*a*a*a * Omegasquared * Sigma / (hR*hR);


    p->ax += dvx * torque * Gamma_r / h;
    p->ay += dvy * torque * Gamma_r / h;
    p->az += dvz * torque * Gamma_r / h;
}

struct reb_vec3d IOPF_unit_T_vector(struct reb_particle* const p, struct reb_particle* primary) {
    double mu,dx,dy,dz,dvx,dvy,dvz,hx,hy,hz,h;

    mu = p->sim->G*(p->m+primary->m);
    dx = p->x - primary->x;
    dy = p->y - primary->y;
    dz = p->z - primary->z;
    dvx = p->vx - primary->vx;
    dvy = p->vy - primary->vy;
    dvz = p->vz - primary->vz;

    hx = (dy*dvz - dz*dvy);
    hy = (dz*dvx - dx*dvz);
    hz = (dx*dvy - dy*dvx);
    h = sqrt(hx*hx + hy*hy + hz*hz);

    double d, ux, uy, uz;
    mag_dir_3d(dx, dy, dz, &d, &ux, &uy, &uz);

    double _v2, uvx, uvy, uvz;
    mag2_dir_3d(dvx, dvy, dvz, &_v2, &uvx, &uvy, &uvz);

    double rv, phi_x, phi_y, phi_z;
    rv = ux * uvx + uy * uvy + uz * uvz,
    phi_x = ux * rv - uvx;
    phi_y = uy * rv - uvy;
    phi_z = uz * rv - uvz;

    struct reb_vec3d T[1];
    mag2_dir_3d(phi_x, phi_y, phi_z, &_v2, &T->x, &T->y, &T->z);

    return *T;
}

void IOPF_torque_jonathan_all(struct reb_simulation* reb_sim) {
    struct reb_particle* const particles = reb_sim->particles;
    const int N = reb_sim->N;

    for (int i = 1; i < N; i++) {
        IOPF_torque_jonathan_force(particles + i, particles);
    }
}

void IOPF_torque_jonathan_force(struct reb_particle* const p, struct reb_particle* primary) {
    double mu,dx,dy,dz,dvx,dvy,dvz,hx,hy,hz,h;

    mu = p->sim->G*(p->m+primary->m);
    dx = p->x - primary->x;
    dy = p->y - primary->y;
    dz = p->z - primary->z;
    dvx = p->vx - primary->vx;
    dvy = p->vy - primary->vy;
    dvz = p->vz - primary->vz;

    hx = (dy*dvz - dz*dvy);
    hy = (dz*dvx - dx*dvz);
    hz = (dx*dvy - dy*dvx);
    h = sqrt(hx*hx + hy*hy + hz*hz);

    double d, ux, uy, uz;
    mag_dir_3d(dx, dy, dz, &d, &ux, &uy, &uz);

    double _v2, uvx, uvy, uvz;
    mag2_dir_3d(dvx, dvy, dvz, &_v2, &uvx, &uvy, &uvz);

    double rv, phi_x, phi_y, phi_z;
    rv = ux * uvx + uy * uvy + uz * uvz,
    phi_x = ux * rv - uvx;
    phi_y = uy * rv - uvy;
    phi_z = uz * rv - uvz;
    mag2_dir_3d(phi_x, phi_y, phi_z, &_v2, &phi_x, &phi_y, &phi_z);

    struct interp_loc iloc = interp_locate(d, STD_PROF_X, STD_PROF_N);

    double Omegasquared, hR, Gamma_r, Sigma, torque, t_m;
    hR = interp_eval(iloc, SCALE_HEIGHT_PROF);
    Sigma = interp_eval(iloc, SURFACE_DENSITY_PROF);
    torque = interp_eval(iloc, TORQUE_PROF);

    Omegasquared = mu / (d*d*d);
    Gamma_r = p->m * d*d*d*d * Omegasquared * Sigma / (hR*hR);

    p->ax += -phi_x * torque * Gamma_r / d;
    p->ay += -phi_y * torque * Gamma_r / d;
    p->az += -phi_z * torque * Gamma_r / d;
}
