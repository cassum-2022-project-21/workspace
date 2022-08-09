#include <rebound.h>

#include "profiles.h"
#include "common.h"
#include "forces.h"
#include "drag_force.h"
#include "migration_torque.h"

void IOPF_drag_torque_all(struct reb_simulation* reb_sim) {
    struct reb_particle* const particles = reb_sim->particles;
    const int N = reb_sim->N;

    for (int i = 1; i < N; i++) {
        IOPF_torque_force(particles + i, particles);
        IOPF_drag_force(particles + i, particles, NULL);
    }
}

void IOPF_drag_all(struct reb_simulation* reb_sim) {
    struct reb_particle* const particles = reb_sim->particles;
    const int N = reb_sim->N;

    for (int i = 1; i < N; i++) {
        IOPF_drag_force(particles + i, particles, NULL);
    }
}
