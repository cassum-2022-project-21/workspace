# %%
from pathlib import Path
import numpy as np
from sim import rebforces
from amuse.units import units

from amuse import TrackLiteratureReferences

TrackLiteratureReferences.suppress_output()
gas_profile_root = Path("disk/calculated_profiles/")
gas_profile_name = "20220721"
gas_profile = np.load(gas_profile_root / gas_profile_name / "all_variables.npz")

# rebforces.set_profiles(
#     len(gas_profile["r"]),
#     gas_profile["r"],
#     gas_profile["velocity"].T * rebforces.CM_PER_S,
#     gas_profile["rho_0"] * rebforces.G_PER_CM3,
#     1.0
# )

# rebforces.copy_np_to_c(gas_profile["sigma"] * rebforces.G_PER_CM3 / 14959787070000, rebforces.SURFACE_DENSITY_PROF, rebforces.STD_PROF_N.value)
# rebforces.copy_np_to_c(gas_profile["H"] / gas_profile["r_cm"], rebforces.SCALE_HEIGHT_PROF, rebforces.STD_PROF_N.value)
# rebforces.copy_np_to_c(gas_profile["torque"], rebforces.TORQUE_PROF, rebforces.STD_PROF_N.value)

# %%
import rebound
from collections import namedtuple

EARTH_MASS = (1 | units.MEarth).value_in(units.MSun)

class TestOutput(namedtuple("TestOutput", ["t", "dt", "a", "rDa", "e", "rDe", "i", "rDi", "h", "Dh"])):
    def __str__(self):
        return (
            "t={t:.5f}  dt={dt:.5f}\n"
            "a={a:.5f} rDa={rDa:+.5e}\n"
            "e={e:.5f} rDe={rDe:+.5e}\n"
            "i={i:.5f} rDi={rDi:+.5e}\n"
            "           Dh={Dh:+.5e}\n\n"
        ).format(**self._asdict())

class TestSimulation(rebound.Simulation):
    def __init__(self, m=EARTH_MASS, rho_p=np.float64(3.) * rebforces.G_PER_CM3, a=0.25, e=0.0, inc=0.0, output_dt=None):
        super().__init__(self)
        
        self.G = 4 * np.pi ** 2
        self.integrator = "mercurius"

        sun = rebound.Particle(m=1.)
        self.add(sun)

        p = rebound.Particle(simulation=self, primary=self.particles[0], m=m, r=(3 * m / (4 * np.pi * rho_p))**(1./3), a=a, e=e, inc=inc)
        self.add(p)

        if output_dt is None:
            self.output_dt = self.particles[1].P
        else:
            self.output_dt = output_dt

        self.old_t = self.t
        self.old_orbit = self.particles[1].calculate_orbit()

    def __iter__(self):
        while True:
            self.integrate(self.t + self.output_dt)

            dt = self.t - self.old_t
            new_orbit = self.particles[1].calculate_orbit()

            def rD(attr):
                val = getattr(new_orbit, attr)
                if val == 0.0:
                    return 0
                else:
                    return (val - getattr(self.old_orbit, attr)) / dt / val

            ret = TestOutput(
                t = self.t,
                dt = dt,
                a = new_orbit.a,
                rDa = rD("a"),
                e = new_orbit.e,
                rDe = rD("e"),
                i = new_orbit.inc,
                rDi = rD("inc"),
                h = new_orbit.h,
                Dh = (new_orbit.h - self.old_orbit.h) / dt
            )

            yield ret

            self.old_t = self.t
            self.old_orbit = new_orbit

# %%
import numpy as np
from sim import rebforces
from amuse.units import units

def set_simple_profile(a, alpha, Sigma, eta, torque, C_D, h, torque_transition=None):
    r = np.linspace(a-0.05, a+0.05, rebforces.__STD_PROF_NMAX__)
    r_cm = r * 14959787070000
    velocity = np.stack([(2 * np.pi / np.sqrt(r)) * (1 - 2 * eta)**2, np.zeros_like(r)])
    Sigma = Sigma * np.exp(-alpha * (r - a))
    h = np.full_like(r, h)
    rho_0 = Sigma / (h * r_cm * np.sqrt(2*np.pi))

    if torque_transition is None:
        torque = np.full_like(r, torque)
    else:
        torque = torque * (2 / (1 + np.exp(1000 * (r - torque_transition))) - 1)

    rebforces.set_profiles(
        rebforces.__STD_PROF_NMAX__,
        r,
        velocity,
        rho_0 * rebforces.G_PER_CM3,
        C_D
    )

# rebforces.set_profiles(
#     len(gas_profile["r"]),
#     gas_profile["r"],
#     gas_profile["velocity"].T * rebforces.CM_PER_S,
#     gas_profile["rho_0"] * rebforces.G_PER_CM3,
#     1.0
# )

    rebforces.copy_np_to_c(Sigma * rebforces.G_PER_CM3 / 14959787070000, rebforces.SURFACE_DENSITY_PROF, rebforces.STD_PROF_N.value)
    rebforces.copy_np_to_c(h, rebforces.SCALE_HEIGHT_PROF, rebforces.STD_PROF_N.value)
    rebforces.copy_np_to_c(torque, rebforces.TORQUE_PROF, rebforces.STD_PROF_N.value)

# # %%
# a = 0.15
# i = np.argmin(np.abs(gas_profile["r"] - a))

# a = gas_profile["r"][i]
# Sigma = gas_profile["sigma"][i]
# torque = 10
# C_D = 0.44
# eta = -0.002
# h = gas_profile["H"][i] / gas_profile["r_cm"][i]

# lnSigma = np.log(gas_profile["sigma"])
# alpha = -np.gradient(lnSigma, gas_profile["r"])[i]

# set_simple_profile(a, alpha, Sigma, eta, torque, C_D, h)
# # set_simple_profile(a, alpha, Sigma, eta, torque, C_D, h, 0.225)

a = 0.21
C_D = 1.0

rebforces.set_profiles(
    len(gas_profile["r"]),
    gas_profile["r"],
    gas_profile["velocity"].T * rebforces.CM_PER_S,
    gas_profile["rho_0"] * rebforces.G_PER_CM3,
    C_D
)

rebforces.copy_np_to_c(gas_profile["sigma"] * rebforces.G_PER_CM3 / 14959787070000, rebforces.SURFACE_DENSITY_PROF, rebforces.STD_PROF_N.value)
rebforces.copy_np_to_c(gas_profile["H"] / gas_profile["r_cm"], rebforces.SCALE_HEIGHT_PROF, rebforces.STD_PROF_N.value)
rebforces.copy_np_to_c(100 * gas_profile["torque"], rebforces.TORQUE_PROF, rebforces.STD_PROF_N.value)

# %%
from itertools import islice
from ctypes import byref

test_sim = TestSimulation(a=a, m=0.1 * EARTH_MASS, e=0.01, inc=0.0, output_dt=1000)

test_sim.additional_forces = rebforces.IOPF_torque_jonathan_all
test_sim.force_is_velocity_dependent = 1

for i in test_sim:
    print(i)

    p = test_sim.particles[1]
    orbit = p.calculate_orbit()

    primary = test_sim.particles[0]
    r_p = (p.x - primary.x, p.y - primary.y, p.z - primary.z)
    r_h = r_p / np.linalg.norm(r_p)
    print(f"r = {r_h}")

    T = rebforces.IOPF_unit_T_vector(byref(p), byref(primary))
    T_p = np.array((T.x, T.y, T.z))
    print(f"T = {T_p}")

    print(f"r . T = {np.dot(r_p, T_p)}")
    print(f"angle = {np.arccos(np.clip(np.dot(r_p, T_p), -1.0, 1.0))}")

    # adachi_vK = 2 * np.pi / np.sqrt(p.a)
    # iloc = rebforces.interp_locate_binary(p.a, rebforces.STD_PROF_X, rebforces.STD_PROF_N)
    # rho_0i = rebforces.interp_eval_cubic(iloc, rebforces.DENSITY_PROF)
    # adachi_tau0 = 2 * p.m / (np.pi * C_D * p.r**2 * adachi_vK * rho_0i)

    # adachi_rDa = -2 * ((0.97 * orbit.e + 0.64 * orbit.inc + eta) * eta + (0.16 * alpha + 0.35) * orbit.e**3 + 0.16 * orbit.inc**2) / adachi_tau0
    # adachi_rDe = -(0.77 * orbit.e + 0.64 * orbit.inc + eta) / adachi_tau0
    # adachi_rDi = -0.5 * (0.77 * orbit.e + 0.85 * orbit.inc + eta) / adachi_tau0

    # print(
    #     "Adachi predictions:\n"
    #     f"rDa={adachi_rDa:+.5e}\n"
    #     f"rDe={adachi_rDe:+.5e}\n"
    #     f"rDi={adachi_rDi:+.5e}\n"
    # )

    # omega_K2 = 4 * np.pi**2 / (orbit.a * orbit.a * orbit.a)
    # Sigma_i = rebforces.interp_eval_cubic(iloc, rebforces.SURFACE_DENSITY_PROF)
    # h_i = rebforces.interp_eval_cubic(iloc, rebforces.SCALE_HEIGHT_PROF)
    # gamma_r = p.m * Sigma_i * orbit.a**4 * omega_K2 / (h_i * h_i)

    # print(
    #     "Torque law:\n"
    #     f"Dh = {gamma_r * torque:+.5e}\n"
    # )