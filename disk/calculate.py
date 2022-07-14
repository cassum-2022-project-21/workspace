import numpy as np
from pathlib import Path
import sys
from scipy.interpolate import interp1d
import numbers

_should_save = False
_output_dir = Path.cwd()

if __name__ == "__main__":
    _should_save = True

    if len(sys.argv) > 2:
        _input_dir = Path(sys.argv[1]).resolve()
        _output_dir = Path("__file__").resolve().parent / "calculated_profiles" / sys.argv[2]
        _output_dir.mkdir(parents=True, exist_ok=True)
    else:
        _input_dir = Path(sys.argv[1]).resolve()

def load_data(name, interp_r=None, interp_kind=None):
    _r, _l = np.loadtxt(_input_dir / f"{name}.txt").T
    if interp_r is not None:
        return interp1d(_r, _l, kind=interp_kind)(interp_r)
    else:
        return _l

def save_data(name, data, **kwargs):
    if _should_save:
        np.savetxt(_output_dir / name, data, **kwargs)

r = np.loadtxt(_input_dir / "rad_chart.txt")
save_data("rad_chart.txt", r, fmt="%.9e")

# Merge velocity files
vt_gas = load_data("vt_gas")
vr_gas = load_data("vr_gas")

velocity = np.column_stack([vt_gas, vr_gas])
save_data("velocity.txt", velocity, fmt="%.9e")

# Calculate density
T = load_data("temperature")
k_B = 1.3807e-16 # cm^2 g s^-2 K^-1
m_p = 1.66e-24; mu = 2.33
gamma = 1.0 # Use new value of gamma = 1.0 not 1.4
c_s = np.sqrt(gamma * k_B * T / (mu * m_p)) # cm s^-1

r_cm = r * 14959787069100
Ω = 2 * np.pi * (r ** (-3.0 / 2)) / (31558150) # s^-1
H = c_s / Ω
v_K = r_cm * Ω

rho_p = 3
v = v_K - vt_gas
va = np.abs(v)
vs = np.sign(v)

sigma = load_data("sigma")
rho_0 = sigma / (H * np.sqrt(2*np.pi))
save_data("midplane_density.txt", np.column_stack([r, rho_0]), fmt="%.9e")

alpha = load_data("alpha")

nd = sigma / (2 * H * m_p * mu)
la = 1 / (nd * 2e-15)
nu = la * c_s
nu_effective = alpha * c_s * H
eta_vis = nu * rho_0

Re_a = 2 * va * rho_0 / eta_vis

lM = np.linspace(-3, 0, 100)
M = np.power(10., lM)
a = (M * 5.97e27 * 3 / (rho_p * 4 * np.pi))**(1. / 3)
Re = Re_a[:, np.newaxis] * a[np.newaxis, :]

C_D = 1.0
F_D = -0.5 * C_D * np.pi * (a**2.)[np.newaxis, :] * (rho_0 * va**2. * vs)[:, np.newaxis]
a_D = F_D / (M * 5.97e27)[np.newaxis, :]
t_fric = -(M * 5.97e27)[np.newaxis, :] * v[:, np.newaxis] / F_D

tau = t_fric * Ω[:, np.newaxis]

eta = 1 - np.square(vt_gas / v_K)
eta_adachi = eta / 2
v_rp = (-eta * v_K)[:, np.newaxis] / (tau + 1. / tau)

if _should_save:
    np.savez_compressed(_output_dir / "all_variables", **{ k: v for k, v in locals().items() if not k.startswith("_") and isinstance(v, np.ndarray) or isinstance(v, numbers.Number) })
