#ifndef IOPF_MIGRATION_TORQUE
#define IOPF_MIGRATION_TORQUE

#include <rebound.h>

void IOPF_torque_all(struct reb_simulation* reb_sim);
void IOPF_torque_force(struct reb_particle* p, struct reb_particle* primary);

void IOPF_torque_jonathan_all(struct reb_simulation* reb_sim);
void IOPF_torque_jonathan_force(struct reb_particle* p, struct reb_particle* primary);


#endif
