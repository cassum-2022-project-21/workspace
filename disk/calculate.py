import numpy as np

# Create velocity.txt and density.txt for the data from Xiao

# Merge velocity files
r, vt_gas = np.loadtxt("mdot9disk20220601/vt_gas.txt").T
_, vr_gas = np.loadtxt("mdot9disk20220601/vr_gas.txt").T

velocity = np.column_stack([r, vt_gas, vr_gas])
np.savetxt("velocity.txt", velocity, fmt="%.6e")

# Calculate density
_, T = np.loadtxt("mdot9disk20220601/temperature.txt").T
k_B = 1.3807e-16 # cm^2 g s^-2 K^-1
m_p = 1.66e-24; mu = 2.33
c_s = np.sqrt(k_B * T / (mu * m_p)) # cm s^-1
Ω = 2 * np.pi * (r ** (-3.0 / 2)) / (31558150) # s^-1
H = c_s / Ω
np.savetxt("scale_height_2.txt", np.column_stack([r, H]), fmt="%.6e")

_, sigma = np.loadtxt("mdot9disk20220601/sigma.txt").T
rho_0 = sigma / (H * np.sqrt(2*np.pi))
np.savetxt("midplane_density_2.txt", np.column_stack([r, rho_0]), fmt="%.6e")
