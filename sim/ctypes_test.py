# %%
import simulation
import rebforces
# %%
sim = simulation.Simulation()

# %%
sim.parse_arguments(
    [
        "-t", "10000.0",
        "-d", "100.0",
        "-n", "100",
        "-c", "rebound",
        "--a-in", "0.22",
        "--a-out", "0.24",
        "--std-i", "0.01",
        "--std-e", "0.02",
        "-m", "1.0",
        "--rho", "3.0",
        "--no-continue"
    ]
)

# %%
sim.init()

# %%
sim.ic_generate()

# %%
import numpy as np
from pathlib import Path

profile_root = Path("../disk/calculated_profiles/")
profile_name = "20220714_1"
profile = np.load(profile_root / profile_name / "all_variables.npz")

# %%
rebforces.set_profiles(
    len(profile["r"]),
    profile["r"],
    profile["velocity"].T * rebforces.CM_PER_S,
    profile["rho_0"] * rebforces.G_PER_CM3,
    profile["C_D"]
)

sim.sim.additional_forces = rebforces.IOPF_drag_all
sim.sim.force_is_velocity_dependent = 1

sim.evolve_model()

