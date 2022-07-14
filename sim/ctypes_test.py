# %%
import simulation
import rebforces
from ctypes import byref

# %%
sim = simulation.Simulation()

# %%
sim.parse_arguments(
    [
        "-t", "10.0",
        "-d", "0.5",
        "-n", "1",
        "-c", "rebound",
        "--a-in", "0.26",
        "--a-out", "0.26",
        "--std-i", "0",
        "--std-e", "0.05",
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
    profile["velocity"].T,
    profile["rho_0"],
    profile["C_D"]
)

sim.sim.additional_forces = rebforces.IOPF_drag_all
sim.sim.force_is_velocity_dependent = 1

sim.evolve_model()

