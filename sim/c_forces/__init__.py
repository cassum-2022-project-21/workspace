from ctypes import CDLL, c_double, c_uint8, POINTER
from pathlib import Path

librebcforces = CDLL(Path(__file__).parent.resolve() / "librebcforces.so")

IOPF_drag_force = librebcforces.IOPF_drag_force
IOPF_drag_force.restype = c_double
IOPF_drag_all = librebcforces.IOPF_drag_all

def _np_to_c(arr):
    return (c_double * len(arr))(*arr)

def set_profiles(r, velocity, density):
    c_uint8.in_dll(librebcforces, "STD_PROF_X_N").value = len(r)
    velocity_prof_c = POINTER(POINTER(c_double)).in_dll(librebcforces, "VELOCITY_PROF")
    velocity_prof_c[0].value = _np_to_c(velocity[0])
    POINTER(c_double).in_dll(librebcforces, "DENSITY_PROF").value = _np_to_c(density)
