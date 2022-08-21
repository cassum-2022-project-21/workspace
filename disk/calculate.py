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
        return interp1d(_r, _l, kind=interp_kind or "linear", fill_value="extrapolate")(interp_r)
    else:
        return _l

def save_data(name, data, **kwargs):
    if _should_save:
        np.savetxt(_output_dir / name, data, **kwargs)

r = np.loadtxt(_input_dir / "rad_chart.txt")
r = np.exp(np.linspace(np.log(r.min()), np.log(r.max()), len(r) * 2))
save_data("rad_chart.txt", r, fmt="%.9e")

# Merge velocity files
vt_gas = load_data("vt_gas", r)
vr_gas = load_data("vr_gas", r)

velocity = np.column_stack([vt_gas, vr_gas])
save_data("velocity.txt", velocity, fmt="%.9e")

# Calculate density
T = load_data("temperature", r)
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

sigma = load_data("sigma", r)
rho_0 = sigma / (H * np.sqrt(2*np.pi))
save_data("midplane_density.txt", np.column_stack([r, rho_0]), fmt="%.9e")

alpha = load_data("alpha", r)

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

# fa = 0.225
# fc = 50
# fm = 0.03
# fk = 640
# fb = 1
# fsr = fk * (r - fa)
# torque = -fb * fsr / (fsr * fsr + fc) + fm 

torque = load_data("torque", r / 2.3)

alpha_grad = (- r / sigma) * np.gradient(sigma, r)
beta_grad = (- r / T) * np.gradient(T, r)
paardekooper_torque = -0.85 - alpha_grad - 0.9 * beta_grad

if _should_save:
    from matplotlib.rcsetup import cycler
    import matplotlib.pyplot as plt

    l = np.logical_and(r > 0.15, r < 0.30)

    plt.rc('axes', prop_cycle=(cycler(marker=['o'], ms=[4])))

    fig, _axs = plt.subplots(2, 3, figsize=(20, 8))
    axs = np.ravel(_axs)

    axs[0].plot(r[l], torque[l])
    axs[0].set_ylabel("Torque [$\Gamma_0$]")

    axs[1].plot(r[l], (H/r_cm)[l])
    axs[1].set_ylim(0, 0.05)
    axs[1].set_ylabel("$h/r$")

    axs[2].plot(r[l], alpha[l])
    axs[2].set_yscale("log")
    axs[2].set_ylim(1e-5, 1e-2)
    axs[2].set_ylabel("$\\alpha$")

    axs[4].plot(r[l], sigma[l])
    axs[4].set_ylabel("$\\Sigma$ [g / cm$^2$]")

    axs[3].hlines(0, 0.15, 0.30, linestyles="--", color="black")
    axs[3].plot(r[l], eta[l])
    axs[3].set_ylim(-0.003, 0.003)
    axs[3].set_ylabel("$\\eta$")
    # axs[3].yaxis.set_major_formatter(tickformat)

    axs[5].plot(r[l], rho_0[l])
    axs[5].set_ylabel("$\\rho_{g,0}$ [g / cm$^3$]")

    plt.savefig(_output_dir / "disk.png")

    np.savez_compressed(_output_dir / "all_variables", **{ k: v for k, v in locals().items() if not k.startswith("_") and isinstance(v, np.ndarray) or isinstance(v, numbers.Number) })
