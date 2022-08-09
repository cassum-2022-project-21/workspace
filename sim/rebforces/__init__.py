from ctypes import CDLL, c_double, c_uint, POINTER, Structure
from pathlib import Path
import rebound
import numpy as np

librebcforces = CDLL(Path(__file__).parent.resolve() / "librebcforces.so")

class IOPFDragForceDiag(Structure):
    _fields_ = [
        ("orbit", rebound.Orbit), # Orbital elements of the planet

        ("vt_gas", c_double), # Tangential velocity of gas at d
        ("vr_gas", c_double), # Radial velocity of gas at d
        ("rho_0", c_double), # Density

        ("vx_gas", c_double), # x velocity of gas
        ("vy_gas", c_double), # y velocity of gas

        ("v_gas", c_double), # Velocity of the gas
        ("v_rel", c_double), # Relative velocity of planet and gas

        ("ux_rel", c_double), # x component of unit normal relative velocity
        ("uy_rel", c_double), # y component of unit normal relative velocity

        ("a_d", c_double), # Acceleration of the drag force
        ("P_d", c_double), # Power of the drag force
    ]

IOPF_drag_force = librebcforces.IOPF_drag_force
IOPF_drag_force.argtypes = [POINTER(rebound.Particle), POINTER(rebound.Particle), POINTER(IOPFDragForceDiag)]
IOPF_drag_force.restype = None

IOPF_torque_force = librebcforces.IOPF_torque_force
IOPF_torque_force.argtypes = [POINTER(rebound.Particle), POINTER(rebound.Particle)]
IOPF_torque_force.restype = None

IOPF_torque_jonathan_force = librebcforces.IOPF_torque_jonathan_force
IOPF_torque_jonathan_force.argtypes = [POINTER(rebound.Particle), POINTER(rebound.Particle)]
IOPF_torque_jonathan_force.restype = None

IOPF_drag_all = librebcforces.IOPF_drag_all
IOPF_drag_all.argtypes = [POINTER(rebound.Simulation)]
IOPF_drag_all.restype = None

IOPF_torque_all = librebcforces.IOPF_torque_all
IOPF_torque_all.argtypes = [POINTER(rebound.Simulation)]
IOPF_torque_all.restype = None

IOPF_torque_jonathan_all = librebcforces.IOPF_torque_jonathan_all
IOPF_torque_jonathan_all.argtypes = [POINTER(rebound.Simulation)]
IOPF_torque_jonathan_all.restype = None

IOPF_drag_torque_all = librebcforces.IOPF_drag_torque_all
IOPF_drag_torque_all.argtypes = [POINTER(rebound.Simulation)]
IOPF_drag_torque_all.restype = None


class InterpLoc(Structure):
    _fields_ = [
        ("s", c_double),
        ("idx", c_uint)
    ]

interp_locate_linear = librebcforces.interp_locate_linear
interp_locate_linear.argtypes = [c_double, POINTER(c_double), c_uint]
interp_locate_linear.restype = InterpLoc

interp_locate_binary = librebcforces.interp_locate_binary
interp_locate_binary.argtypes = [c_double, POINTER(c_double), c_uint]
interp_locate_binary.restype = InterpLoc

interp_eval = librebcforces.interp_eval
interp_eval.argtypes = [InterpLoc, POINTER(c_double)]
interp_eval.restype = c_double

interp_eval_cubic = librebcforces.interp_eval_cubic
interp_eval_cubic.argtypes = [InterpLoc, POINTER(c_double)]
interp_eval_cubic.restype = c_double

__STD_PROF_NMAX__ = 2048 # This should match the value in profiles.h
STD_PROF_N = c_uint.in_dll(librebcforces, "STD_PROF_N")
STD_PROF_X = (c_double * __STD_PROF_NMAX__).in_dll(librebcforces, "STD_PROF_X")
VELOCITY_PROF = ((c_double * __STD_PROF_NMAX__) * 2).in_dll(librebcforces, "VELOCITY_PROF")
DENSITY_PROF = (c_double * __STD_PROF_NMAX__).in_dll(librebcforces, "DENSITY_PROF")
DRAG_COEFF = c_double.in_dll(librebcforces, "DRAG_COEFF")
SURFACE_DENSITY_PROF = (c_double * __STD_PROF_NMAX__).in_dll(librebcforces, "SURFACE_DENSITY_PROF")
SCALE_HEIGHT_PROF = (c_double * __STD_PROF_NMAX__).in_dll(librebcforces, "SCALE_HEIGHT_PROF")
TORQUE_PROF = (c_double * __STD_PROF_NMAX__).in_dll(librebcforces, "TORQUE_PROF")


CM_PER_S = c_double.in_dll(librebcforces, "CM_PER_S")
G_PER_CM3 = c_double.in_dll(librebcforces, "G_PER_CM3")

def copy_np_to_c(np_arr, dest, n):
    if np_arr.ndim != 1: # Doesn't support converting numpy 2D array to C 2D array yet
        raise NotImplementedError("Arrays with greater than 1 dimension are not supported")
    np.ctypeslib.as_array(dest)[0:n] = np_arr[0:n]

def set_profiles(n, r, velocity, density, C_D):
    if n < 1 or n > __STD_PROF_NMAX__:
        raise ValueError(f"n must be positive and less than or equal to {__STD_PROF_NMAX__}")

    STD_PROF_N.value = n
    copy_np_to_c(r, STD_PROF_X, STD_PROF_N.value)
    copy_np_to_c(velocity[0], VELOCITY_PROF[0], STD_PROF_N.value)
    copy_np_to_c(velocity[1], VELOCITY_PROF[1], STD_PROF_N.value)
    copy_np_to_c(density, DENSITY_PROF, STD_PROF_N.value)
    DRAG_COEFF.value = C_D
