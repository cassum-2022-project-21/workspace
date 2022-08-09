#ifndef IOPF_FORCES
#define IOPF_FORCES

#include <rebound.h>

#include "drag_force.h"

void IOPF_drag_torque_all(struct reb_simulation* reb_sim);
void IOPF_drag_all(struct reb_simulation* reb_sim);

#endif
