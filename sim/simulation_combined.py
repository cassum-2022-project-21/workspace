__author__ = "Maxwell Cai"

# from abie import ABIE
from lib2to3.pgen2.literals import simple_escapes
from data_io import DataIO
import rebound
import numpy as np
from amuse.units import units
from amuse.ic.salpeter import new_powerlaw_mass_distribution
import matplotlib.pyplot as plt
import argparse
import os
from scipy.interpolate import interp1d
import time
from pathlib import Path
import rebforces

def mag_dir_2d(x, y):
    r = np.sqrt(x*x + y*y)
    return r, x / r, y / r

gas_profile_root = Path(__file__).parent / "../disk/calculated_profiles/"
gas_profile_name = "20200820"
gas_profile = np.load(gas_profile_root / gas_profile_name / "all_variables.npz")

rebforces.set_profiles(
    len(gas_profile["r"]),
    gas_profile["r"],
    gas_profile["velocity"].T * rebforces.CM_PER_S,
    gas_profile["rho_0"] * rebforces.G_PER_CM3,
    0.0
)

rebforces.copy_np_to_c(gas_profile["sigma"] * rebforces.G_PER_CM3 / 14959787070000, rebforces.SURFACE_DENSITY_PROF, rebforces.STD_PROF_N.value)
rebforces.copy_np_to_c(gas_profile["H"] / gas_profile["r_cm"], rebforces.SCALE_HEIGHT_PROF, rebforces.STD_PROF_N.value)
rebforces.copy_np_to_c(gas_profile["torque"], rebforces.TORQUE_PROF, rebforces.STD_PROF_N.value)

class Simulation(object):

    def __init__(self, sim=None, verbose=True, io=True):
        self.args = None
        self.sim = sim
        self.buffer_rebound = None
        self.early_stop_time = None
        self.pa_beta_f = None
        self.reb_continue = False

        self.verbose = True
        self.io = io

    def parse_arguments(self, override=None):
        parser = argparse.ArgumentParser()
        parser.add_argument('-s', '--random-seed', dest='seed', type=int, default=0, help='random seed')
        parser.add_argument('-t', '--t_end', dest='t_end', type=float, default=10.0, help='Termination time')
        parser.add_argument('--N_end', dest='N_end', type=int, default=-1, help='# of particles for early termination')
        parser.add_argument('--N_enddelay', dest='N_enddelay', type=float, default=10000.0, help='Delay on early termination [yr]')
        parser.add_argument('-d', '--store-dt', dest='store_dt', type=float, default=1.0, help='Snapshot time')
        parser.add_argument('-n', '--n-particles', dest='n_p', type=int, default=100, help='number of planetesimals')
        parser.add_argument('-c', '--code-name', dest='code', type=str, default=None, help='N-body code name')
        parser.add_argument('--a-in', dest='a_in', type=float, default=0.1, help='Inner semi-major axis')
        parser.add_argument('--a-out', dest='a_out', type=float, default=0.12, help='Inner semi-major axis')
        parser.add_argument('--alpha', dest='alpha', default=None, help='Planetesimal mass function power index')
        parser.add_argument('--std-i', dest='std_i', default=0.01, type=float, help='Standard derivation of inclination (rad)')
        parser.add_argument('--std-e', dest='std_e', default=0.02, type=float, help='Standard deviation of eccentricity')
        parser.add_argument('-m', '--m-total', dest='m_total', type=float, default=1.0, help='Planetesimal total mass [MEarth]')
        parser.add_argument('-M', '--m-star', dest='m_star', type=float, default=1.0, help='host star mass [MSun]')
        parser.add_argument('--rho', dest='rho', type=float, default=2.0, help='Density of planetesimals [g/cm^3]')
        parser.add_argument('--ef', dest='ef', type=float, default=1.0, help='Radii enlargement factor')
        parser.add_argument('--planet-mass', dest='pm', type=float, default=0.0, help='The mass of the inner super-Earth. If 0, the super-Earth will not be included.[MEarth]')
        parser.add_argument('--planet-a', dest='pa', type=float, default=0.1, help='The semi-major axis of the inner super-Earth. [AU]')
        parser.add_argument('--planet-rh', dest='prh', type=float, default=10.0, help='Separation between the planet and the planetesimal rings [Hill radii]')
        parser.add_argument('--planet-rho', dest='prho', type=float, default=5.0, help='The mean density of the planet [g/cm^3]')
        parser.add_argument('--pa-rate', dest='pa_rate', type=float, default=0.0, help='The pebble accretion rate [MSun/yr]')
        parser.add_argument('--pa-beta', dest='pa_beta', type=str, default="2_3", help='The mass-dependent pebble accretion parameter')

        parser.add_argument('--rebound-archive', dest='rebound_archive', type=str, default=None, help='A rebound archive to save to / read from (with --continue)')
        parser.add_argument('--no-continue', dest='reb_no_continue', action='store_true', help="Prevent continuing from a rebound snapshot archive")
        parser.add_argument('--continue-from', dest='reb_continue_from', type=float, default=-1.0, help="Time to continue from")

        parser.add_argument('-C', '--drag-coefficient', dest="C_d", type=float, default=0.0, help="The drag coefficient C_d")
        parser.add_argument('--velocity-file', nargs="?", dest="velocity_file", type=str, const="velocity.txt", default=None)
        parser.add_argument('--density-file', nargs="?", dest="density_file", type=str, const="density.txt", default=None)

        parser.add_argument('--migration-torque', dest='migration_torque', action='store_true', help="Add a migration torque force")

        parser.add_argument('--N_handoff', dest='N_handoff', type=int, default=-1, help="Switch to mercurius integrator")

        if override is None:
            self.args = parser.parse_args()
        else:
            self.args = parser.parse_args(override)

        n, d = self.args.pa_beta.split("_")
        self.pa_beta_f = float(n) / float(d)

        if self.args.rebound_archive is None:
            self.args.rebound_archive = f"rebound_archive_{self.args.seed}.bin"

    def init(self):
        self.print('Using rebound as the integrator...')
        if not self.args.reb_no_continue and os.path.isfile(self.args.rebound_archive):
            self.print('Able to continue from rebound archive', self.args.rebound_archive)
            self.reb_continue = True

        if self.reb_continue:
            archive = rebound.SimulationArchive(self.args.rebound_archive)
            self.sim = archive.getSimulation(self.args.reb_continue_from if self.args.reb_continue_from >= 0.0 else archive.tmax)
        else:
            self.sim = rebound.Simulation()
        self.sim.integrator = 'IAS15'
        self.sim.G = 4 * np.pi ** 2.0  # AU/MSun/yr
        self.sim.collision = 'direct'
        self.sim.collision_resolve = 'merge'

        if self.io:
            self.buffer_rebound = DataIO(output_file_name='data_reb_%d.h5' % self.args.seed, CONST_G=self.sim.G, append=True)

        self.print(self.args)

    def init_mercurius(self):
        self.sim.integrator = 'mercurius'

        self.sim.dt = 0.001

        self.sim.ri_mercurius.hillfac = 5
        self.sim.ri_mercurius.safe_mode = False

        self.sim.ri_ias15.min_dt = 0

        self.sim.ri_whfast.safe_mode = False
        self.sim.ri_whfast.coordinates = "democraticheliocentric"

        self.sim.collision = 'direct'
        self.sim.collision_resolve = 'merge'


    def ic_generate(self):
        if self.reb_continue:
            raise RuntimeError("Generating ICs when continuing in rebound is prohibited. Use ic_continue() instead.")
    
        np.random.seed(self.args.seed)
        # alpha = 0: random and uniform distribution of mass; alpha < 0: power-law IMF; alpha = None: equal-mass
        if self.args.alpha is None:
            self.print('Creating equal-mass system...')
            m_p = np.ones(self.args.n_p) * (self.args.m_total / self.args.n_p) | units.MEarth
        else:
            self.print('Creating system with a mass spectrum of alpha = %f' % float(self.args.alpha))
            m_p = new_powerlaw_mass_distribution(self.args.n_p, 0.003|units.MEarth, 0.1|units.MEarth, alpha=float(self.args.alpha))
            # scaling to the total mass of planetesimals
            m_scale_factor = m_p.value_in(units.MEarth).sum()
            m_p /= m_scale_factor
            self.print('Total mass = %f [MEarth] (scaled)' % m_p.value_in(units.MEarth).sum())
        a_p = np.random.uniform(low=self.args.a_in, high=self.args.a_out, size=(self.args.n_p, ))
        # The std(i) = 0.01 and std(e) = 0.02 of Rayleigh distribution is adopted from Kokubo & Ida (2002), section 3.1
        e_rayleigh_scale = self.args.std_e / np.sqrt(0.429)  # the constant 0.429 see Wikipedia
        i_rayleigh_scale = self.args.std_i / np.sqrt(0.429)  # the constant 0.429 see Wikipedia
        e_p = np.random.rayleigh(scale=e_rayleigh_scale, size=self.args.n_p)
        i_p = np.random.rayleigh(scale=i_rayleigh_scale, size=self.args.n_p)
        f_p = np.random.uniform(low=0.0, high=2*np.pi, size=(self.args.n_p, ))
        rho = self.args.rho | (units.g / units.cm ** 3)
        r_p = (self.args.ef * (0.75 * m_p / rho / np.pi) ** (1. / 3)).value_in(units.AU)

        if self.args.pm == 0:
            a_gap = 0.0
            r_planet = 0.0
        elif self.args.pm > 0:
            a_gap = self.args.pa * ((self.args.pm|units.MEarth).value_in(units.MSun) / (3 * self.args.m_star)) ** (1.0 / 3) * self.args.prh
            self.print(a_gap)
            rho_planet = self.args.prho | (units.g / units.cm ** 3)
            r_planet = ((0.75 * (self.args.pm|units.MEarth) / rho_planet / np.pi) ** (1. / 3)).value_in(units.AU)
        else:
            a_gap = 0.0
            r_planet = 0.0


        self.print('Adding the host star of %f Solar masses' % self.args.m_star)
        self.sim.add(m=self.args.m_star, x=0, y=0, z=0, vx=0, vy=0, vz=0, r=0.00465,
                        hash=np.random.randint(100000000, 999999999))
        if self.args.pm > 0:
            self.print('Adding a planet at a=%f with m=%f Earth masses' % (self.args.pa, self.args.pm))
            self.sim.add(m=self.args.pm, a=self.args.pa, r=r_planet, primary=self.sim.particles[0],
                            hash=np.random.randint(100000000, 999999999))
        self.print('Adding N=%d planetesimals...' % self.args.n_p)
        for i in range(len(a_p)):
            self.sim.add(m=m_p[i].value_in(units.MSun), a=a_p[i]+a_gap, e=e_p[i], inc=i_p[i], f=f_p[i], r=r_p[i],
                            primary=self.sim.particles[0], hash=np.random.randint(100000000, 999999999))
        self.sim.move_to_com()
        # initialize the buffer

        if self.io:
            self.buffer_rebound.initialize_buffer(self.sim.N)
            lim = 0.2
            fig, _ = rebound.OrbitPlot(self.sim, color=True, unitlabel="[AU]", xlim=(-lim, lim), ylim=(-lim, lim))
            plt.savefig('orbits.pdf')
            plt.close(fig)

        self.add_drag_force()
    
    def ic_continue(self):
        self.add_drag_force()
        self.buffer_rebound.initialize_buffer(self.sim.N)

    def add_drag_force(self):
        if self.args.C_d != 0.0:
            rebforces.DRAG_COEFF.value = self.args.C_d
            
            if self.args.migration_torque:
                self.print("Enabling migration torque and drag force")
                self.sim.additional_forces = rebforces.IOPF_drag_torque_all
            else:
                self.print("Enabling drag force")
                self.sim.additional_forces = rebforces.IOPF_drag_all
            
            self.sim.force_is_velocity_dependent = 1
        elif self.args.migration_torque:
            self.print("Enabling migration torque")
            self.sim.additional_forces = rebforces.IOPF_torque_all
            self.sim.force_is_velocity_dependent = 1

            # C_d = self.args.C_d

            # r, vt_gas_cms, vr_gas_cms = np.loadtxt(self.args.velocity_file).T
            # vt_gas = (vt_gas_cms | (units.cm / units.s)).value_in(units.AU / units.yr)
            # vr_gas = (vr_gas_cms | (units.cm / units.s)).value_in(units.AU / units.yr)

            # _, rho_0_cms = np.loadtxt(self.args.density_file).T
            # rho_0 = (rho_0_cms | (units.g / units.cm**3)).value_in(units.MSun / (units.AU**3))
            
            # interpf = interp1d(r, np.stack([vt_gas, vr_gas, rho_0]))

            # def drag_force(reb_sim):
            #     for p in self.sim.particles[1:]:
            #         x = p.x
            #         y = p.y
            #         # z = p.z

            #         _r, ux, uy = mag_dir_2d(x, y)

            #         _vt_gas, _vr_gas, _rho_0 = interpf(_r)

            #         vx_rel = ux * _vr_gas - uy * _vt_gas - p.vx
            #         vy_rel = uy * _vr_gas + ux * _vt_gas - p.vy
            #         v_rel, ux_rel, uy_rel = mag_dir_2d(vx_rel, vy_rel)

            #         A = np.pi * p.r * p.r
            #         F_d = 0.5 * _rho_0 * v_rel * v_rel * C_d * A

            #         p.ax += F_d * ux_rel / p.m
            #         p.ay += F_d * uy_rel / p.m


    def store_hdf5_rebound(self, energy):
        self.sim.simulationarchive_snapshot(self.args.rebound_archive)

        if self.sim.N < self.buffer_rebound.buf_x.shape[1]:
            self.buffer_rebound.flush()
            self.buffer_rebound.reset_buffer()
            self.buffer_rebound.initialize_buffer(self.sim.N)
        # Special routine to store rebound data into ABIE HDF5 format
        x = np.full(self.sim.N, np.nan)
        y = np.full(self.sim.N, np.nan)
        z = np.full(self.sim.N, np.nan)
        vx = np.full(self.sim.N, np.nan)
        vy = np.full(self.sim.N, np.nan)
        vz = np.full(self.sim.N, np.nan)
        masses = np.full(self.sim.N, np.nan)
        semi = np.full(self.sim.N, np.nan)
        ecc = np.full(self.sim.N, np.nan)
        inc = np.full(self.sim.N, np.nan)
        hashes = np.full(self.sim.N, -1, dtype=int)
        orbits = self.sim.calculate_orbits()
        for i in range(self.sim.N):
            x[i] = self.sim.particles[i].x
            y[i] = self.sim.particles[i].y
            z[i] = self.sim.particles[i].z
            vx[i] = self.sim.particles[i].vx
            vy[i] = self.sim.particles[i].vy
            vz[i] = self.sim.particles[i].vz
            masses[i] = self.sim.particles[i].m
            hashes[i] = self.sim.particles[i].hash.value
            if i > 0:
                semi[i] = orbits[i - 1].a
                ecc[i] = orbits[i - 1].e
                inc[i] = orbits[i - 1].inc
        pos = np.column_stack([x, y, z]).flatten()
        vel = np.column_stack([vx, vy, vz]).flatten()
        self.buffer_rebound.store_state(self.sim.t, pos=pos, vel=vel, masses=masses, a=semi, e=ecc, i=inc, names=hashes,
                                        energy=energy)

    def pebble_accretion(self):
        if self.args.pa_rate == 0:
            return
        else:
            pa_rate = self.args.pa_rate * self.dt
            m_i2 = np.zeros(self.sim.N)
            for i in range(1, self.sim.N):
                m_i2[i] = self.sim.particles[i].m ** self.pa_beta_f
            mtot = m_i2.sum()
            for i in range(1, self.sim.N):
                self.sim.particles[i].m += (m_i2[i]/mtot*pa_rate)

    def evolve_model(self):
        self.sim.exact_finish_time = 0
        self.print('Start integration...')
        self.evolve_start_t = time.time()

        self.print('t = %f, N = %d, sysT = %f' % (self.sim.t, self.sim.N, time.time() - self.evolve_start_t))
        e_init = self.sim.calculate_energy()
        self.store_hdf5_rebound(e_init)
        self.t_store = self.sim.t
        while self.sim.t < self.args.t_end and not self.should_early_stop:
            while self.sim.t < self.t_store + self.dt:
                try:
                    self.sim.integrate(self.t_store+self.dt)
                    self.pebble_accretion()
                    self.sim.move_to_com()
                except rebound.Collision as error:
                    self.print('A collision occurred', error)
                    # store state in the event of collision
                    e = self.sim.calculate_energy()
                    de = abs((e-e_init)/e_init)
                    self.print('t = %f, N = %d, sysT = %f, dE/E = %e' % (self.sim.t, self.sim.N, time.time() - self.evolve_start_t, de))
                    self.store_hdf5_rebound(e)
                    # update early stopping condition
                    self.reset_early_stop()
            self.t_store = self.sim.t
            e = self.sim.calculate_energy()
            de = abs((e-e_init)/e_init)
            self.print('t = %f, N = %d, sysT = %f, dE/E = %e' % (self.sim.t, self.sim.N, time.time() - self.evolve_start_t, de))
            self.print(self.sim.particles[1].a)
            self.store_hdf5_rebound(e)

            if self.sim.N <= self.args.N_handoff:
                self.sim.integrator_synchronize()

                print(f"N <= {self.args.N_handoff}: Switching to mercurius")
                self.init_mercurius()

                self.args.N_handoff = -1

    def finalize(self):
        if self.io:
            self.buffer_rebound.close()
    
    def save(self):
        if self.io:
            e = self.sim.calculate_energy()
            self.store_hdf5_rebound(e)

    def reset_early_stop(self):
        if self.sim.N <= self.args.N_end:
            self.early_stop_time = self.sim.t + self.args.N_enddelay
            self.print(f"Early stopping by t={self.early_stop_time} barring no further collisions")

    def print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    @property
    def should_early_stop(self):
        return self.early_stop_time is not None and self.sim.t > self.early_stop_time

    @property
    def dt(self):
        return self.args.store_dt

if __name__ == "__main__":
    import signal
    import sys

    sim = Simulation()
    sim.parse_arguments()
    sim.init()

    if sim.reb_continue:
        sim.ic_continue()
    else:
        sim.ic_generate()

    def interrupt_handler(signum, frame):
        print(f"Interrupt at {sim.sim.t=}. Calling sim.finalize()", file=sys.stderr)
        sim.finalize()
        sys.exit(1)

    signal.signal(signal.SIGINT, interrupt_handler)

    sim.evolve_model()

    try:
        sim.save()
        sim.finalize()
        open("DONE", "w").close()
    except KeyboardInterrupt:
        interrupt_handler(None, None)
