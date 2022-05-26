__author__ = "Maxwell Cai"

# from abie import ABIE
# from data_io import DataIO
import rebound
import numpy as np
from amuse.lab import *
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse


class Simulation(object):

    def __init__(self):
        self.args = None
        self.sim = None
        self.buffer_rebound = None

    def parse_arguments(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-s', '--random-seed', dest='seed', type=int, default=0, help='random seed')
        parser.add_argument('-t', '--t_end', dest='t_end', type=float, default=10.0, help='Termination time')
        parser.add_argument('-d', '--store-dt', dest='store_dt', type=float, default=1.0, help='Snapshot time')
        parser.add_argument('-n', '--n-particles', dest='n_p', type=int, default=100, help='number of planetesimals')
        parser.add_argument('-c', '--code-name', dest='code', type=str, default=None, help='N-body code name')
        parser.add_argument('--a-in', dest='a_in', type=float, default=0.1, help='Inner semi-major axis')
        parser.add_argument('--a-out', dest='a_out', type=float, default=0.12, help='Inner semi-major axis')
        parser.add_argument('--alpha', dest='alpha', default=None, help='Planetesimal mass function power index')
        parser.add_argument('--std-i', dest='std_i', default=0.01, type=float, help='Standard derivation of inclination (rad)')
        parser.add_argument('--std-e', dest='std_e', default=0.02, type=float, help='Standard deviation of eccentricity')
        parser.add_argument('-m', '-m-total', dest='m_total', type=float, default=1.0, help='Planetesimal total mass [MEarth]')
        parser.add_argument('-M', '-m-star', dest='m_star', type=float, default=1.0, help='host star mass [MSun]')
        parser.add_argument('--rho', dest='rho', type=float, default=2.0, help='Density of planetesimals [g/cm^3]')
        parser.add_argument('--ef', dest='ef', type=float, default=1.0, help='Radii enlargement factor')
        parser.add_argument('--planet-mass', dest='pm', type=float, default=0.0, help='The mass of the inner super-Earth. If 0, the super-Earth will not be included.[MEarth]')
        parser.add_argument('--planet-a', dest='pa', type=float, default=0.1, help='The semi-major axis of the inner super-Earth. [AU]')
        parser.add_argument('--planet-rh', dest='prh', type=float, default=10.0, help='Separation between the planet and the planetesimal rings [Hill radii]')
        parser.add_argument('--planet-rho', dest='prho', type=float, default=5.0, help='The mean density of the planet [g/cm^3]')
        parser.add_argument('--pa-rate', dest='pa_rate', type=float, default=0.0, help='The pebble accretion rate [MSun/yr]')

        self.args = parser.parse_args()

    def init(self):
        if self.args.code == 'abie':
            print('Using ABIE as the integrator...')
            self.sim = ABIE()
            self.sim.integrator = 'GaussRadau15'
            self.sim.CONST_G = 4 * np.pi ** 2.0  # AU/MSun/yr
            self.sim.store_dt = self.args.store_dt
            self.sim.output_file = 'data_%d.h5' % self.args.seed

        else:
            print('Using rebound as the integrator...')
            self.sim = rebound.Simulation()
            self.sim.integrator = 'IAS15'
            self.sim.G = 4 * np.pi ** 2.0  # AU/MSun/yr
            self.sim.collision = 'direct'
            self.sim.collision_resolve = 'merge'
            # make use of ABIE data buffer
            self.buffer_rebound = DataIO(output_file_name='data_reb_%d.h5' % self.args.seed, CONST_G=self.sim.G)
        print(self.args)


    def ic_generate(self):
        np.random.seed(self.args.seed)
        # alpha = 0: random and uniform distribution of mass; alpha < 0: power-law IMF; alpha = None: equal-mass
        if self.args.alpha is None:
            print('Creating equal-mass system...')
            m_p = np.ones(self.args.n_p) * (self.args.m_total / self.args.n_p) | units.MEarth
        else:
            print('Creating system with a mass spectrum of alpha = %f' % float(self.args.alpha))
            m_p = new_powerlaw_mass_distribution(self.args.n_p, 0.003|units.MEarth, 0.1|units.MEarth, alpha=float(self.args.alpha))
            # scaling to the total mass of planetesimals
            m_scale_factor = m_p.value_in(units.MEarth).sum()
            m_p /= m_scale_factor
            print('Total mass = %f [MEarth] (scaled)' % m_p.value_in(units.MEarth).sum())
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
            print(a_gap)
            rho_planet = self.args.prho | (units.g / units.cm ** 3)
            r_planet = ((0.75 * (self.args.pm|units.MEarth) / rho_planet / np.pi) ** (1. / 3)).value_in(units.AU)
        else:
            a_gap = 0.0
            r_planet = 0.0

        if self.args.code == 'abie':
            print('Adding the host star of %f Solar masses' % self.args.m_star)
            self.sim.add(mass=self.args.m_star, x=0, y=0, z=0, vx=0, vy=0, vz=0, radius=0.00465, name='Sun')
            if self.args.pm > 0:
                print('Adding a planet at a=%f with m=%f Earth masses' % (self.args.pa, self.args.pm))
                self.sim.add(mass=self.args.pm, a=self.args.pa, radius=r_planet, name='Planet', primary='Sun')
            print('Adding N=%d planetesimals...' % self.args.n_p)
            for i in range(len(a_p)):
                self.sim.add(mass=m_p[i].value_in(units.MSun), a=a_p[i]+a_gap, e=e_p[i], i=i_p[i], f=f_p[i], radius=r_p[i],
                             name=('p_%d' % i), primary='Sun')
        else:
            print('Adding the host star of %f Solar masses' % self.args.m_star)
            self.sim.add(m=self.args.m_star, x=0, y=0, z=0, vx=0, vy=0, vz=0, r=0.00465,
                         hash=np.random.randint(100000000, 999999999))
            if self.args.pm > 0:
                print('Adding a planet at a=%f with m=%f Earth masses' % (self.args.pa, self.args.pm))
                self.sim.add(m=self.args.pm, a=self.args.pa, r=r_planet, primary=self.sim.particles[0],
                             hash=np.random.randint(100000000, 999999999))
            print('Adding N=%d planetesimals...' % self.args.n_p)
            for i in range(len(a_p)):
                self.sim.add(m=m_p[i].value_in(units.MSun), a=a_p[i]+a_gap, e=e_p[i], inc=i_p[i], f=f_p[i], r=r_p[i],
                             primary=self.sim.particles[0], hash=np.random.randint(100000000, 999999999))
            self.sim.move_to_com()
            # initialize the buffer
            self.buffer_rebound.initialize_buffer(self.sim.N)
            fig = rebound.OrbitPlot(self.sim, color=True, unitlabel="[AU]", lim=2)
            plt.savefig('orbits.pdf')
            plt.close(fig)


    def store_hdf5_rebound(self ,energy):
        if self.sim.N < self.buffer_rebound.buf_x.shape[1]:
            self.buffer_rebound.flush()
            self.buffer_rebound.reset_buffer()
            self.buffer_rebound.initialize_buffer(self.sim.N)
        # Special routine to store rebound data into ABIE HDF5 format
        x = np.zeros(self.sim.N) * np.nan
        y = np.zeros(self.sim.N) * np.nan
        z = np.zeros(self.sim.N) * np.nan
        vx = np.zeros(self.sim.N) * np.nan
        vy = np.zeros(self.sim.N) * np.nan
        vz = np.zeros(self.sim.N) * np.nan
        masses = np.zeros(self.sim.N) * np.nan
        semi = np.zeros(self.sim.N) * np.nan
        ecc = np.zeros(self.sim.N) * np.nan
        inc = np.zeros(self.sim.N) * np.nan
        hashes = np.zeros(self.sim.N, dtype=np.int) * np.nan
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
            pa_rate = self.args.pa_rate * self.args.store_dt
            m_i2 = np.zeros(self.sim.N)
            for i in range(1, self.sim.N):
                # m_i2[i] = self.sim.particles[i].m ** (4./3)
                m_i2[i] = self.sim.particles[i].m ** (2./3)
            mtot = m_i2.sum()
            for i in range(1, self.sim.N):
                self.sim.particles[i].m += (m_i2[i]/mtot*pa_rate)



    def evolve_model(self):
        print('Start integration...')
        if self.args.code == 'abie':
            self.sim.integrate(self.args.t_end)
        else:
            e_init = self.sim.calculate_energy()
            t_store = self.sim.t
            while self.sim.t < self.args.t_end:
                while self.sim.t < t_store + self.args.store_dt:
                    try:
                        self.sim.integrate(t_store+self.args.store_dt)
                        self.pebble_accretion()
                    except rebound.Collision as error:
                        print('A collision occurred', error)
                        # store state in the event of collision
                        e = self.sim.calculate_energy()
                        de = abs((e-e_init)/e_init)
                        print('t = %f, N = %d, dE/E = %e' % (self.sim.t, self.sim.N, de))
                        self.store_hdf5_rebound(e)
                t_store += self.args.store_dt
                e = self.sim.calculate_energy()
                de = abs((e-e_init)/e_init)
                print('t = %f, N = %d, dE/E = %e' % (self.sim.t, self.sim.N, de))
                self.store_hdf5_rebound(e)

    def finalize(self):
        if self.args.code == 'abie':
            self.sim.stop()
        else:
            self.buffer_rebound.close()

if __name__ == "__main__":
    sim = Simulation()
    sim.parse_arguments()
    sim.init()
    sim.ic_generate()
    sim.evolve_model()
    sim.finalize()