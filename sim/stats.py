import numpy as np

def pprops(sim):
    ps = sim.particles

    m = np.empty(sim.N-1)
    a = np.empty(sim.N-1)
    e = np.empty(sim.N-1)
    inc = np.empty(sim.N-1)
    P = np.empty(sim.N-1)

    for i in range(1, sim.N):
        m[i-1] = ps[i].m
        a[i-1] = ps[i].a
        e[i-1] = ps[i].e
        inc[i-1] = ps[i].inc
        P[i-1] = ps[i].P

    return m, a, e, inc, P
