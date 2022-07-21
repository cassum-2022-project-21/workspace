import matplotlib.pyplot as plt

for k, v in sims.items():
    if v["n_particles"] > 50:
        runtimes = v["runtimes"].T

        Ns = []
        for N in np.arange(runtimes[0, 0], runtimes[0, -1] - 1, -1):
            Ns.append((N - 1, np.mean(runtimes[1, runtimes[0] == N])))
        Ns = np.array(Ns).T
        plt.plot(Ns[0], Ns[1], label=k)

plt.yscale("log")
plt.ylabel("Simulation Years / Second of runtime")
plt.legend()
plt.show()

# %%
from pathlib import Path
from util import NUMBER_REGEX
import numpy as np
import pandas as pd

simon_dir = "sim/simon_simulations_20220714_1/"
simon_path = Path(simon_dir)

# # %%
sims = {}

for sim_path in simon_path.glob("iopf*"):
    label = sim_path.name
    slabel = label.split("_")
    drag_coefficient = int(slabel[slabel.index('DRAG')+1][0])
    n_particles = int(slabel[slabel.index('N')+1])
    seed = int(slabel[-1])

    output_path = sim_path / "output.txt"
    with open(output_path, "r") as output_file:
        _t = 0
        _sysT = 0
        L = []
        for line in output_file:
            if line.startswith("t = "):
                t, N, sysT, *_ = map(float, NUMBER_REGEX.findall(line))

                if sysT > _sysT:
                    dt = t - _t
                    dsysT = sysT - _sysT
                    L.append((N, dt/sysT))

                _t = t
                _sysT = sysT
    
    sims[(drag_coefficient, n_particles, seed)] = {
        "label": label,
        "drag_coefficient": drag_coefficient,
        "n_particles": n_particles,
        "seed": seed,
        "path": sim_path,
        "runtimes": np.array(L)
    }

    print(f"{label} {drag_coefficient=} {n_particles=} {seed=}")
    print()

# %%
import matplotlib.pyplot as plt

for k, v in sims.items():
    if v["n_particles"] > 50:
        runtimes = v["runtimes"].T

        Ns = []
        for N in np.arange(runtimes[0, 0], runtimes[0, -1] - 1, -1):
            Ns.append((N - 1, np.mean(runtimes[1, runtimes[0] == N])))
        Ns = np.array(Ns).T
        plt.plot(Ns[0], Ns[1], label=k)

plt.yscale("log")
plt.ylabel("Simulation Years / Second of runtime")
plt.legend()
plt.show()

# %%
import matplotlib.pyplot as plt

for k, v in sims.items():
    if v["n_particles"] > 50:
        runtimes = v["runtimes"].T

        Ns = []
        for N in np.arange(runtimes[0, 0], runtimes[0, -1] - 1, -1):
            Ns.append((N - 1, np.median(runtimes[1, runtimes[0] == N])))
        Ns = np.array(Ns).T
        plt.plot(Ns[0], Ns[1], label=k)

plt.yscale("log")
plt.ylabel("Simulation Years / Second of runtime")
plt.legend()
plt.show()

# %%
import matplotlib.pyplot as plt

for k, v in sims.items():
    if v["n_particles"] > 50:
        runtimes = v["runtimes"].T

        Ns = []
        for N in np.arange(runtimes[0, 0], runtimes[0, -1] - 1, -1):
            Ns.append((N - 1, np.mean(runtimes[1, runtimes[0] == N])))
        Ns = np.array(Ns).T
        plt.plot(Ns[0], Ns[1], label=k)

plt.yscale("log")
plt.ylabel("Simulation Years / Second of runtime")
plt.legend()
plt.show()

# %%
sims = {}

c = 0
def p(*args, **kwargs):
    if c < 10:
        print(*args, **kwargs)
        c += 1

for sim_path in simon_path.glob("iopf*"):
    label = sim_path.name
    slabel = label.split("_")
    drag_coefficient = int(slabel[slabel.index('DRAG')+1][0])
    n_particles = int(slabel[slabel.index('N')+1])
    seed = int(slabel[-1])

    output_path = sim_path / "output.txt"
    with open(output_path, "r") as output_file:
        _t = 0
        _sysT = 0
        L = []
        for line in output_file:
            if line.startswith("t = "):
                t, N, sysT, *_ = map(float, NUMBER_REGEX.findall(line))

                if sysT > _sysT:
                    dt = t - _t
                    dsysT = sysT - _sysT
                    L.append((N, dt/sysT))

                p(t, N, sysT, _t, _sysT, dt/sysT)

                _t = t
                _sysT = sysT
    
    sims[(drag_coefficient, n_particles, seed)] = {
        "label": label,
        "drag_coefficient": drag_coefficient,
        "n_particles": n_particles,
        "seed": seed,
        "path": sim_path,
        "runtimes": np.array(L)
    }

    print(f"{label} {drag_coefficient=} {n_particles=} {seed=}")
    print()

# %%
sims = {}

c = 0
def p(*args, **kwargs):
    global c
    if c < 10:
        print(*args, **kwargs)
        c += 1

for sim_path in simon_path.glob("iopf*"):
    label = sim_path.name
    slabel = label.split("_")
    drag_coefficient = int(slabel[slabel.index('DRAG')+1][0])
    n_particles = int(slabel[slabel.index('N')+1])
    seed = int(slabel[-1])

    output_path = sim_path / "output.txt"
    with open(output_path, "r") as output_file:
        _t = 0
        _sysT = 0
        L = []
        for line in output_file:
            if line.startswith("t = "):
                t, N, sysT, *_ = map(float, NUMBER_REGEX.findall(line))

                if sysT > _sysT:
                    dt = t - _t
                    dsysT = sysT - _sysT
                    L.append((N, dt/sysT))

                p(t, N, sysT, _t, _sysT, dt/sysT)

                _t = t
                _sysT = sysT
    
    sims[(drag_coefficient, n_particles, seed)] = {
        "label": label,
        "drag_coefficient": drag_coefficient,
        "n_particles": n_particles,
        "seed": seed,
        "path": sim_path,
        "runtimes": np.array(L)
    }

    print(f"{label} {drag_coefficient=} {n_particles=} {seed=}")
    print()

# %%
sims = {}

c = 0
def p(*args, **kwargs):
    global c
    if c < 10:
        print(*args, **kwargs)
        c += 1

for sim_path in simon_path.glob("iopf*"):
    label = sim_path.name
    slabel = label.split("_")
    drag_coefficient = int(slabel[slabel.index('DRAG')+1][0])
    n_particles = int(slabel[slabel.index('N')+1])
    seed = int(slabel[-1])

    output_path = sim_path / "output.txt"
    with open(output_path, "r") as output_file:
        _t = 0
        _sysT = 0
        L = []
        for line in output_file:
            if line.startswith("t = "):
                t, N, sysT, *_ = map(float, NUMBER_REGEX.findall(line))

                if sysT > _sysT:
                    dt = t - _t
                    dsysT = sysT - _sysT
                    L.append((N, dt/sysT))

                p(t, N, sysT, dt, dsysT, dt/sysT)

                _t = t
                _sysT = sysT
    
    sims[(drag_coefficient, n_particles, seed)] = {
        "label": label,
        "drag_coefficient": drag_coefficient,
        "n_particles": n_particles,
        "seed": seed,
        "path": sim_path,
        "runtimes": np.array(L)
    }

    print(f"{label} {drag_coefficient=} {n_particles=} {seed=}")
    print()

# %%
sims = {}

c = 0
def p(*args, **kwargs):
    global c
    if c < 10:
        print(*args, **kwargs)
        c += 1

for sim_path in simon_path.glob("iopf*"):
    label = sim_path.name
    slabel = label.split("_")
    drag_coefficient = int(slabel[slabel.index('DRAG')+1][0])
    n_particles = int(slabel[slabel.index('N')+1])
    seed = int(slabel[-1])

    output_path = sim_path / "output.txt"
    with open(output_path, "r") as output_file:
        _t = 0
        _sysT = 0
        L = []
        for line in output_file:
            if line.startswith("t = "):
                t, N, sysT, *_ = map(float, NUMBER_REGEX.findall(line))

                if sysT > _sysT:
                    dt = t - _t
                    dsysT = sysT - _sysT
                    L.append((N, dt/dsysT))

                _t = t
                _sysT = sysT
    
    sims[(drag_coefficient, n_particles, seed)] = {
        "label": label,
        "drag_coefficient": drag_coefficient,
        "n_particles": n_particles,
        "seed": seed,
        "path": sim_path,
        "runtimes": np.array(L)
    }

    print(f"{label} {drag_coefficient=} {n_particles=} {seed=}")
    print()

# %%
import matplotlib.pyplot as plt

for k, v in sims.items():
    if v["n_particles"] > 50:
        runtimes = v["runtimes"].T

        Ns = []
        for N in np.arange(runtimes[0, 0], runtimes[0, -1] - 1, -1):
            Ns.append((N - 1, np.mean(runtimes[1, runtimes[0] == N])))
        Ns = np.array(Ns).T
        plt.plot(Ns[0], Ns[1], label=k)

plt.yscale("log")
plt.ylabel("Simulation Years / Second of runtime")
plt.legend()
plt.show()

# %%
import matplotlib.pyplot as plt

for k, v in sims.items():
    if v["n_particles"] < 25:
        runtimes = v["runtimes"].T

        Ns = []
        for N in np.arange(runtimes[0, 0], runtimes[0, -1] - 1, -1):
            Ns.append((N - 1, np.mean(runtimes[1, runtimes[0] == N])))
        Ns = np.array(Ns).T
        plt.plot(Ns[0], Ns[1], label=k)

plt.yscale("log")
plt.ylabel("Simulation Years / Second of runtime")
plt.legend()
plt.show()

# %%
import matplotlib.pyplot as plt

for k, v in sims.items():
    if v["n_particles"] < 50:
        runtimes = v["runtimes"].T

        Ns = []
        for N in np.arange(runtimes[0, 0], runtimes[0, -1] - 1, -1):
            Ns.append((N - 1, np.mean(runtimes[1, runtimes[0] == N])))
        Ns = np.array(Ns).T
        plt.plot(Ns[0], Ns[1], label=k)

plt.yscale("log")
plt.ylabel("Simulation Years / Second of runtime")
plt.legend()
plt.show()

# %%
import matplotlib.pyplot as plt

for k, v in sims.items():
    if v["n_particles"] < 50:
        runtimes = v["runtimes"].T

        Ns = []
        for N in np.arange(runtimes[0, 0], runtimes[0, -1] - 1, -1):
            Ns.append((N - 1, np.mean(runtimes[1, runtimes[0] == N])))
        Ns = np.array(Ns).T
        plt.plot(Ns[0], Ns[1], label=k)

# plt.yscale("log")
plt.ylabel("Simulation Years / Second of runtime")
plt.legend()
plt.show()

# %%
import matplotlib.pyplot as plt

for k, v in sims.items():
    if v["n_particles"] > 50:
        runtimes = v["runtimes"].T

        Ns = []
        for N in np.arange(runtimes[0, 0], runtimes[0, -1] - 1, -1):
            Ns.append((N - 1, np.mean(runtimes[1, runtimes[0] == N])))
        Ns = np.array(Ns).T
        plt.plot(Ns[0], Ns[1], label=k)

# plt.yscale("log")
plt.ylabel("Simulation Years / Second of runtime")
plt.legend()
plt.show()

# %%
import matplotlib.pyplot as plt

for k, v in sims.items():
    if v["n_particles"] > 50:
        runtimes = v["runtimes"].T

        Ns = []
        for N in np.arange(runtimes[0, 0], runtimes[0, -1] - 1, -1):
            Ns.append((N - 1, np.mean(runtimes[1, runtimes[0] == N])))
        Ns = np.array(Ns).T
        plt.plot(Ns[0], Ns[1], label=k)

plt.yscale("log")
plt.ylabel("Simulation Years / Second of runtime")
plt.legend()
plt.show()

# %%
import matplotlib.pyplot as plt

for k, v in sims.items():
    if v["n_particles"] > 50:
        runtimes = v["runtimes"].T

        Ns = []
        for N in np.arange(runtimes[0, 0], runtimes[0, -1] - 1, -1):
            Ns.append((N - 1, np.mean(runtimes[1, runtimes[0] == N])))
        Ns = np.array(Ns).T
        plt.plot(Ns[0], Ns[1], label=k)

plt.yscale("log")
plt.ylabel("Simulation Years / Second of runtime")
plt.legend()
plt.show()

# %%
import matplotlib.pyplot as plt

for k, v in sims.items():
    if v["n_particles"] > 50:
        runtimes = v["runtimes"].T

        Ns = []
        for N in np.arange(runtimes[0, 0], runtimes[0, -1] - 1, -1):
            Ns.append((N - 1, np.mean(runtimes[1, runtimes[0] == N])))
        Ns = np.array(Ns).T
        plt.plot(Ns[0], Ns[1], label=k)

plt.yscale("log")
plt.ylabel("Simulation Years / Second of runtime")
plt.xlabel("Planets remaining")
plt.legend()
plt.show()

# %%
import matplotlib.pyplot as plt

for k, v in sims.items():
    if v["n_particles"] > 50:
        runtimes = v["runtimes"].T

        Ns = []
        for N in np.arange(runtimes[0, 0], runtimes[0, -1] - 1, -1):
            Ns.append((N - 1, np.mean(runtimes[1, runtimes[0] == N])))
        Ns = np.array(Ns).T
        plt.plot(Ns[0], Ns[1], label=k)

plt.yscale("log")
plt.ylabel("Simulation Years / Second of runtime")
plt.xlabel("Planets remaining")
plt.xlim(100, 15)
plt.legend()
plt.show()

# %%
import matplotlib.pyplot as plt

for k, v in sims.items():
    if v["n_particles"] > 50 and drag_coefficient == 1:
        runtimes = v["runtimes"].T

        Ns = []
        for N in np.arange(runtimes[0, 0], runtimes[0, -1] - 1, -1):
            Ns.append((N - 1, np.mean(runtimes[1, runtimes[0] == N])))
        Ns = np.array(Ns).T
        plt.plot(Ns[0], Ns[1], label=k)

plt.ylabel("Simulation Years / Second of runtime")
plt.legend()
plt.show()

# %%
import matplotlib.pyplot as plt

for k, v in sims.items():
    if v["n_particles"] > 50 and v["drag_coefficient"] == 0:
        runtimes = v["runtimes"].T
        runtimes2 = sims[(1, k[1], k[2])]["runtimes"].T

        Ns = []
        for N in np.arange(runtimes[0, 0], runtimes[0, -1] - 1, -1):
            Ns.append((N - 1, np.mean(runtimes[1, runtimes[0] == N]) / np.mean(runtimes2[1, runtimes2[0] == N])))
        Ns = np.array(Ns).T
        plt.plot(Ns[0], Ns[1], label=k)

plt.ylabel("Runtime speed ratio")
plt.legend()
plt.show()

# %%
import matplotlib.pyplot as plt

for k, v in sims.items():
    if v["n_particles"] > 50 and v["drag_coefficient"] == 0:
        runtimes = v["runtimes"].T
        runtimes2 = sims[(1, k[1], k[2])]["runtimes"].T

        Ns = []
        for N in np.arange(runtimes[0, 0], runtimes[0, -1] - 1, -1):
            Ns.append((N - 1, np.mean(runtimes[1, runtimes[0] == N]) / np.mean(runtimes2[1, runtimes2[0] == N])))
        Ns = np.array(Ns).T
        plt.plot(Ns[0], Ns[1], label=k)

plt.ylabel("Runtime speed ratio")
plt.xlabel("Planets remaining")
plt.legend()
plt.show()

# %%
import matplotlib.pyplot as plt

for k, v in sims.items():
    if v["n_particles"] > 50 and v["drag_coefficient"] == 0:
        runtimes = v["runtimes"].T
        runtimes2 = sims[(1, k[1], k[2])]["runtimes"].T

        Ns = []
        for N in np.arange(runtimes[0, 0], runtimes[0, -1] - 1, -1):
            Ns.append((N - 1, np.median(runtimes[1, runtimes[0] == N]) / np.median(runtimes2[1, runtimes2[0] == N])))
        Ns = np.array(Ns).T
        plt.plot(Ns[0], Ns[1], label=k)

plt.ylabel("Runtime speed ratio")
plt.xlabel("Planets remaining")
plt.legend()
plt.show()

# %%
import matplotlib.pyplot as plt

for k, v in sims.items():
    if v["n_particles"] < 50 and v["drag_coefficient"] == 0:
        runtimes = v["runtimes"].T
        runtimes2 = sims[(1, k[1], k[2])]["runtimes"].T

        Ns = []
        for N in np.arange(runtimes[0, 0], runtimes[0, -1] - 1, -1):
            Ns.append((N - 1, np.median(runtimes[1, runtimes[0] == N]) / np.median(runtimes2[1, runtimes2[0] == N])))
        Ns = np.array(Ns).T
        plt.plot(Ns[0], Ns[1], label=k)

plt.ylabel("Runtime speed ratio")
plt.xlabel("Planets remaining")
plt.legend()
plt.show()

# %%
import matplotlib.pyplot as plt

for k, v in sims.items():
    if v["n_particles"] > 50 and v["drag_coefficient"] == 0:
        runtimes = v["runtimes"].T
        runtimes2 = sims[(1, k[1], k[2])]["runtimes"].T

        Ns = []
        for N in np.arange(runtimes[0, 0], runtimes[0, -1] - 1, -1):
            Ns.append((N - 1, np.median(runtimes[1, runtimes[0] == N]) / np.median(runtimes2[1, runtimes2[0] == N])))
        Ns = np.array(Ns).T
        plt.plot(Ns[0], Ns[1], label=k)

plt.ylabel("Runtime speed ratio")
plt.xlabel("Planets remaining")
plt.legend()
plt.show()

# %%
import h5py

# %%
from pathlib import Path
from util import NUMBER_REGEX, sorted_steps
import numpy as np
import pandas as pd

simon_dir = "sim/simon_simulations_20220714_1/"
simon_path = Path(simon_dir)

# %%
import h5py
from util import sorted_steps, prop_table

for k, v in sims.items():
    if v["n_particles"] > 50 and v["drag_coefficient"] == 0:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")

        sorted_steps = sorted_steps(f)
        hashes = sorted_steps[0]["hash"][0]
        prop_table = prop_table(hashes)

# %%
import h5py
import util

for k, v in sims.items():
    if v["n_particles"] > 50 and v["drag_coefficient"] == 0:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")

        sorted_steps = util.sorted_steps(f)
        hashes = sorted_steps[0]["hash"][0]
        prop_table = util.prop_table(hashes)

# %%
import h5py
import util

for k, v in sims.items():
    if v["n_particles"] > 50 and v["drag_coefficient"] == 0:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")

        sorted_steps = util.sorted_steps(f)
        hashes = sorted_steps[0]["hash"][0]
        prop_table = util.prop_table(hashes)

        print(hashes)

# %%
import h5py
import util

for k, v in sims.items():
    if v["n_particles"] > 50 and v["drag_coefficient"] == 0:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")

        sorted_steps = util.sorted_steps(f)
        hashes = sorted_steps[0]["hash"][0]
        prop_table = util.prop_table(hashes)

        print(hashes.shape)

# %%
import h5py
import util

for k, v in sims.items():
    if v["n_particles"] > 50 and v["drag_coefficient"] == 0:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")

        sorted_steps = util.sorted_steps(f)
        hashes = sorted_steps[0]["hash"][0]
        prop_table = util.prop_table(hashes)

        print(prop_table)

# %%
import h5py
import util

for k, v in sims.items():
    if v["n_particles"] > 50 and v["drag_coefficient"] == 0:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")

        sorted_steps = util.sorted_steps(f)
        sorted_steps

# %%
import h5py
import util

for k, v in sims.items():
    if v["n_particles"] > 50 and v["drag_coefficient"] == 0:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")

        sorted_steps = util.sorted_steps(f)
        print(sorted_steps)

# %%
import h5py
import util

# def plot_eccentricity_spread_over_time(ax, steps, **kwargs):
#     std_e = []
#     time = []
#     for step in steps:
#         time.extend(step["time"])
#         std_e.extend(step["eccentricity"])
#     ax.plot(time, mass, **kwargs)

for k, v in sims.items():
    if v["n_particles"] > 50 and v["drag_coefficient"] == 0:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")

        sorted_steps = util.sorted_steps(f)
        print(sorted_steps[0])

# %%
import h5py
import util

# def plot_eccentricity_spread_over_time(ax, steps, **kwargs):
#     std_e = []
#     time = []
#     for step in steps:
#         time.extend(step["time"])
#         std_e.extend(step["eccentricity"])
#     ax.plot(time, mass, **kwargs)

for k, v in sims.items():
    if v["n_particles"] > 50 and v["drag_coefficient"] == 0:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")

        sorted_steps = util.sorted_steps(f)
        print(sorted_steps[0].keys())

# %%
import h5py
import util

# def plot_eccentricity_spread_over_time(ax, steps, **kwargs):
#     std_e = []
#     time = []
#     for step in steps:
#         time.extend(step["time"])
#         std_e.extend(step["eccentricity"])
#     ax.plot(time, mass, **kwargs)

for k, v in sims.items():
    if v["n_particles"] > 50 and v["drag_coefficient"] == 0:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")

        sorted_steps = util.sorted_steps(f)
        print(sorted_steps[0]["mass"])

# %%
import h5py
import util

def plot_eccentricity_spread_over_time(ax, steps, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step["ecc"], axis=1))
    ax.plot(time, std_e, **kwargs)

fig, ax = plt.subplots()

for k, v in sims.items():
    if v["n_particles"] > 50 and v["drag_coefficient"] == 0:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        plot_eccentricity_spread_over_time(ax, sorted_steps)
        

# %%
import h5py
import util

def plot_eccentricity_spread_over_time(ax, steps, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step["ecc"], axis=1))
    print(std_e, time)
    ax.plot(time, std_e, **kwargs)

fig, ax = plt.subplots()

for k, v in sims.items():
    if v["n_particles"] > 50 and v["drag_coefficient"] == 0:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        plot_eccentricity_spread_over_time(ax, sorted_steps)
        

# %%
import h5py
import util

k = {"a": None}

def plot_eccentricity_spread_over_time(ax, steps, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step["ecc"], axis=1))
    print(std_e, time)
    ax.plot(time, std_e, **kwargs)

fig, ax = plt.subplots()

for k, v in sims.items():
    if v["n_particles"] > 50 and v["drag_coefficient"] == 0:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        k["a"] = sorted_steps
        plot_eccentricity_spread_over_time(ax, sorted_steps)
        

# %%
import h5py
import util

b = {"a": None}

def plot_eccentricity_spread_over_time(ax, steps, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step["ecc"], axis=1))
    print(std_e, time)
    ax.plot(time, std_e, **kwargs)

fig, ax = plt.subplots()

for k, v in sims.items():
    if v["n_particles"] > 50 and v["drag_coefficient"] == 0:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        b["a"] = sorted_steps
        plot_eccentricity_spread_over_time(ax, sorted_steps)
        

# %%
b["a"][0]["ecc"]

# %%
b["a"][0]["ecc"][0]

# %%
b["a"][0]["ecc"][1]

# %%
import h5py
import util


def plot_eccentricity_spread_over_time(ax, steps, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step["ecc"], axis=1))
    print(std_e, time)
    ax.plot(time, std_e, **kwargs)

fig, ax = plt.subplots()

for k, v in sims.items():
    if v["n_particles"] > 50 and v["drag_coefficient"] == 0:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        plot_eccentricity_spread_over_time(ax, sorted_steps)
        

# %%
import h5py
import util


def plot_eccentricity_spread_over_time(ax, steps, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step["ecc"], axis=1))
    ax.plot(time, std_e, **kwargs)

fig, ax = plt.subplots()

for k, v in sims.items():
    if v["n_particles"] > 50 and v["drag_coefficient"] == 0:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        plot_eccentricity_spread_over_time(ax, sorted_steps)
        

# %%
b["a"][0]["ecc"][1]

# %%
b["a"][0]["mass"][1]

# %%
import h5py
import util


def plot_eccentricity_spread_over_time(ax, steps, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step["ecc"][:, 1:], axis=1))
    ax.plot(time, std_e, **kwargs)

fig, ax = plt.subplots()

for k, v in sims.items():
    if v["n_particles"] > 50 and v["drag_coefficient"] == 0:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        plot_eccentricity_spread_over_time(ax, sorted_steps)
        

# %%
import h5py
import util


def plot_eccentricity_spread_over_time(ax, steps, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step["ecc"][:, 1:], axis=1))
    ax.plot(time, std_e, **kwargs)

fig, ax = plt.subplots()

for k, v in sims.items():
    if v["n_particles"] > 50 and v["drag_coefficient"] == 1:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        plot_eccentricity_spread_over_time(ax, sorted_steps)
        

# %%
import h5py
import util


def plot_eccentricity_spread_over_time(ax, steps, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step["ecc"][:, 1:], axis=1))
    ax.plot(time, std_e, **kwargs)

fig, ax = plt.subplots()

for k, v in sims.items():
    if v["n_particles"] > 50 and v["drag_coefficient"] == 1:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        plot_eccentricity_spread_over_time(ax, sorted_steps, label=k)

ax.legend()        

# %%
import h5py
import util


def plot_eccentricity_spread_over_time(ax, steps, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step["ecc"][:, 1:], axis=1))
    ax.plot(time, std_e, **kwargs)

fig, ax = plt.subplots()

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        plot_eccentricity_spread_over_time(ax, sorted_steps, label=k)

ax.legend()        

# %%
import h5py
import util


def plot_eccentricity_spread_over_time(ax, steps, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step["ecc"][:, 1:], axis=1))
        mean_e.extend(np.mean(step["ecc"][:, 1:], axis=1))
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, **kwargs)

fig, ax = plt.subplots()

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        plot_eccentricity_spread_over_time(ax, sorted_steps, label=k)

ax.legend()        

# %%
import h5py
import util


def plot_eccentricity_spread_over_time(ax, steps, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step["ecc"][:, 1:], axis=1))
        mean_e.extend(np.mean(step["ecc"][:, 1:], axis=1))
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, **kwargs, alpha=0.2)

fig, ax = plt.subplots()

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        plot_eccentricity_spread_over_time(ax, sorted_steps, label=k)

ax.legend()        

# %%
import h5py
import util


def plot_eccentricity_spread_over_time(ax, steps, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step["ecc"][:, 1:], axis=1))
        mean_e.extend(np.mean(step["ecc"][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs, alpha=0.2)
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, **kwargs, alpha=0.2)

fig, ax = plt.subplots()

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        plot_eccentricity_spread_over_time(ax, sorted_steps, label=k)

ax.legend()        

# %%
import h5py
import util


def plot_eccentricity_spread_over_time(ax, steps, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step["ecc"][:, 1:], axis=1))
        mean_e.extend(np.mean(step["ecc"][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs, alpha=0.2)
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, alpha=0.2)

fig, ax = plt.subplots()

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        plot_eccentricity_spread_over_time(ax, sorted_steps, label=k)

ax.legend()        

# %%
import h5py
import util


def plot_eccentricity_spread_over_time(ax, steps, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step["ecc"][:, 1:], axis=1))
        mean_e.extend(np.mean(step["ecc"][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, alpha=0.2)

fig, ax = plt.subplots()

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        plot_eccentricity_spread_over_time(ax, sorted_steps, label=k)

ax.legend()        

# %%
import h5py
import util


def plot_eccentricity_spread_over_time(ax, steps, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step["ecc"][:, 1:], axis=1))
        mean_e.extend(np.mean(step["ecc"][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, alpha=0.2)

fig, ax = plt.subplots()

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        plot_eccentricity_spread_over_time(ax, sorted_steps)

ax.legend()        

# %%
import h5py
import util


def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_eccentricity_spread_over_time(ax, steps, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step["ecc"][:, 1:], axis=1))
        mean_e.extend(np.mean(step["ecc"][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, alpha=0.2)

fig, ax = plt.subplots()

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        plot_eccentricity_spread_over_time(ax, sorted_steps, label=k)

ax.legend()        

# %%
import h5py
import util


def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_eccentricity_spread_over_time(ax, steps, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step["ecc"][:, 1:], axis=1))
        mean_e.extend(np.mean(step["ecc"][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, alpha=0.2)

fig, ax = plt.subplots()

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        plot_eccentricity_spread_over_time(ax, sorted_steps, label=make_label(k))

ax.legend()        

# %%
import h5py
import util


def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_eccentricity_spread_over_time(ax, steps, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step["ecc"][:, 1:], axis=1))
        mean_e.extend(np.mean(step["ecc"][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, alpha=0.2)

fig, ax = plt.subplots(figsize=(10, 8))

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        plot_eccentricity_spread_over_time(ax, sorted_steps, label=make_label(k))

ax.legend()        

# %%
import h5py
import util


def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_eccentricity_spread_over_time(ax, steps, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step["ecc"][:, 1:], axis=1))
        mean_e.extend(np.mean(step["ecc"][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, alpha=0.2, ec="face")

fig, ax = plt.subplots(figsize=(10, 8))

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        plot_eccentricity_spread_over_time(ax, sorted_steps, label=make_label(k))

ax.legend()        

# %%
import h5py
import util


def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_eccentricity_spread_over_time(ax, steps, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step["ecc"][:, 1:], axis=1))
        mean_e.extend(np.mean(step["ecc"][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, alpha=0.2, ec="face")

fig, ax = plt.subplots(figsize=(10, 8))

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        plot_eccentricity_spread_over_time(ax, sorted_steps, label=make_label(k))

ax.legend()
ax.title("Eccentricity (line is mean, shaded is +/- 1 $\\sigma$")

# %%
import h5py
import util


def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_eccentricity_spread_over_time(ax, steps, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step["ecc"][:, 1:], axis=1))
        mean_e.extend(np.mean(step["ecc"][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, alpha=0.2, ec="face")

fig, ax = plt.subplots(figsize=(10, 8))

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        plot_eccentricity_spread_over_time(ax, sorted_steps, label=make_label(k))

ax.legend()
ax.set_title("Eccentricity (line is mean, shaded is +/- 1 $\\sigma$)")

# %%
import h5py
import util


def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_eccentricity_spread_over_time(ax, steps, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step["ecc"][:, 1:], axis=1))
        mean_e.extend(np.mean(step["ecc"][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, alpha=0.2, ec="face")

fig, ax = plt.subplots(figsize=(8, 6))

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        plot_eccentricity_spread_over_time(ax, sorted_steps, label=make_label(k))

ax.legend()
ax.set_title("Eccentricity (line is mean, shaded is +/- 1 $\\sigma$)")

# %%
import h5py
import util


def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_eccentricity_spread_over_time(ax, steps, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step["ecc"][:, 1:], axis=1))
        mean_e.extend(np.mean(step["ecc"][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, alpha=0.2, ec="face")

fig, ax = plt.subplots(figsize=(8, 6))

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        plot_eccentricity_spread_over_time(ax, sorted_steps, label=make_label(k))

ax.legend()
ax.set_title("Eccentricity (line is mean, shaded is +/- 1 stdev$)")

# %%
import h5py
import util


def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_eccentricity_spread_over_time(ax, steps, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step["ecc"][:, 1:], axis=1))
        mean_e.extend(np.mean(step["ecc"][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, alpha=0.2, ec="face")

fig, ax = plt.subplots(figsize=(8, 6))

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        plot_eccentricity_spread_over_time(ax, sorted_steps, label=make_label(k))

ax.legend()
ax.set_title("Eccentricity (line is mean, shaded is +/- 1 stdev)")

# %%
import h5py
import util


def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_eccentricity_spread_over_time(ax, steps, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step["ecc"][:, 1:], axis=1))
        mean_e.extend(np.mean(step["ecc"][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, alpha=0.2, ec="face")

fig, ax = plt.subplots(figsize=(8, 6))

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        plot_eccentricity_spread_over_time(ax, sorted_steps, label=make_label(k))

ax.legend()
ax.set_title("Eccentricity (line is mean, shaded is \\pm 1 stdev)")

# %%
import h5py
import util


def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_eccentricity_spread_over_time(ax, steps, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step["ecc"][:, 1:], axis=1))
        mean_e.extend(np.mean(step["ecc"][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, alpha=0.2, ec="face")

fig, ax = plt.subplots(figsize=(8, 6))

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        plot_eccentricity_spread_over_time(ax, sorted_steps, label=make_label(k))

ax.legend()
ax.set_title("Eccentricity (line is mean, shaded is $\\pm$ 1 stdev)")

# %%
import h5py
import util


def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_eccentricity_spread_over_time(ax, steps, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step["ecc"][:, 1:], axis=1))
        mean_e.extend(np.mean(step["ecc"][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, alpha=0.2, ec="face")

fig, ax = plt.subplots(figsize=(8, 6))

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        plot_eccentricity_spread_over_time(ax, sorted_steps, label=make_label(k))

ax.legend()
ax.set_title("Eccentricity (line is mean, shaded is $\\pm 1$ stdev)")

# %%
import h5py
import util


def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_eccentricity_spread_over_time(ax, steps, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step["ecc"][:, 1:], axis=1))
        mean_e.extend(np.mean(step["ecc"][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, alpha=0.2, ec="face")

fig, ax = plt.subplots(figsize=(8, 6))

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        plot_eccentricity_spread_over_time(ax, sorted_steps, label=make_label(k))

ax.legend()
ax.set_xlim(0, 100000)
ax.set_title("Eccentricity (line is mean, shaded is $\\pm 1$ stdev)")

# %%
import h5py
import util


def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_eccentricity_spread_over_time(ax, steps, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step["ecc"][:, 1:], axis=1))
        mean_e.extend(np.mean(step["ecc"][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, alpha=0.2, ec="face")

fig, axs = plt.subplots(1, 2, figsize=(16, 6))

ax = axs[0]

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        plot_eccentricity_spread_over_time(ax, sorted_steps, label=make_label(k))

ax.legend()
ax.set_xlim(0, 100000)
ax.set_title("Eccentricity (line is mean, shaded is $\\pm 1$ stdev)")

# %%
import h5py
import util


def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_eccentricity_spread_over_time(ax, steps, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step["ecc"][:, 1:], axis=1))
        mean_e.extend(np.mean(step["ecc"][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, alpha=0.2, ec="face")

fig, axs = plt.subplots(1, 2, figsize=(16, 6))

ax = axs[0]

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        plot_eccentricity_spread_over_time(ax, sorted_steps, label=make_label(k))

ax.legend()
ax.set_xlim(0, 100000)
ax.set_title(r"$\bar{e}$")

# %%
import h5py
import util


def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_eccentricity_spread_over_time(ax, steps, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step["ecc"][:, 1:], axis=1))
        mean_e.extend(np.mean(step["ecc"][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, alpha=0.2, ec="face")

fig, axs = plt.subplots(1, 2, figsize=(16, 6))

ax = axs[0]

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        plot_eccentricity_spread_over_time(ax, sorted_steps, label=make_label(k))

ax.legend()
ax.set_xlim(0, 100000)
ax.set_title(r"$\mathrm{mean}(e) \pm st$")

# %%
import h5py
import util


def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_eccentricity_spread_over_time(ax, steps, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step["ecc"][:, 1:], axis=1))
        mean_e.extend(np.mean(step["ecc"][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, alpha=0.2, ec="face")

fig, axs = plt.subplots(1, 2, figsize=(16, 6))

ax = axs[0]

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        plot_eccentricity_spread_over_time(ax, sorted_steps, label=make_label(k))

ax.legend()
ax.set_xlim(0, 100000)
ax.set_title(r"$\mathrm{mean}(e) \pm \mathrm{std}(e)$")

# %%
import h5py
import util


def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_eccentricity_spread_over_time(ax, steps, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step["ecc"][:, 1:], axis=1))
        mean_e.extend(np.mean(step["ecc"][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, alpha=0.2, ec="face")

fig, axs = plt.subplots(1, 2, figsize=(16, 6))

ax = axs[0]

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        plot_eccentricity_spread_over_time(ax, sorted_steps, label=make_label(k))

ax.legend()
ax.set_xlim(0, 100000)
ax.set_title(r"$\mathrm{mean}(e) \pm \mathrm{std}(e)$")
ax.set_xlabel("Time (yr)")

# %%
import h5py
import util


def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_eccentricity_over_time(ax, steps, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step["ecc"][:, 1:], axis=1))
        mean_e.extend(np.mean(step["ecc"][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, alpha=0.2, ec="face")

def plot_std_eccentricity_over_time(ax, steps, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step["ecc"][:, 1:], axis=1))
    ax.plot(time, np.array(std_e), **kwargs)


fig, axs = plt.subplots(1, 2, figsize=(16, 6))

ax = axs[0]

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        plot_eccentricity_over_time(ax, sorted_steps, label=make_label(k))

ax.legend()
ax.set_xlim(0, 100000)
ax.set_title(r"$\mathrm{mean}(e) \pm \mathrm{std}(e)$")
ax.set_xlabel("Time (yr)")

# %%
import h5py
import util


def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_eccentricity_over_time(ax, steps, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step["ecc"][:, 1:], axis=1))
        mean_e.extend(np.mean(step["ecc"][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, alpha=0.2, ec="face")

def plot_std_eccentricity_over_time(ax, steps, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step["ecc"][:, 1:], axis=1))
    ax.plot(time, np.array(std_e), **kwargs)


fig, axs = plt.subplots(1, 2, figsize=(16, 6))

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        plot_eccentricity_over_time(axs[0], sorted_steps, label=make_label(k))
        plot_std_eccentricity_over_time(axs[1], sorted_steps, label=make_label(k))


axs[0].legend()
axs[0].set_xlim(0, 100000)
axs[0].set_title(r"$\mathrm{mean}(e) \pm \mathrm{std}(e)$")
axs[0].set_xlabel("Time (yr)")

axs[1].legend()
axs[1].set_xlim(0, 100000)
axs[1].set_title(r"$\mathrm{std}(e)$")
axs[1].set_xlabel("Time (yr)")

# %%
import h5py
import util


def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_eccentricity_over_time(ax, steps, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step["ecc"][:, 1:], axis=1))
        mean_e.extend(np.mean(step["ecc"][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, alpha=0.2, ec="face")

def plot_std_eccentricity_over_time(ax, steps, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step["ecc"][:, 1:], axis=1))
    ax.plot(time, np.array(std_e), **kwargs)


fig, axs = plt.subplots(1, 2, figsize=(16, 6))

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        plot_eccentricity_over_time(axs[0], sorted_steps, label=make_label(k))
        plot_std_eccentricity_over_time(axs[1], sorted_steps, label=make_label(k))


axs[0].legend()
axs[0].set_xlim(0, 100000)
axs[0].set_title(r"$\mathrm{mean}(e) \pm \mathrm{std}(e)$")
axs[0].set_xlabel("Time (yr)")

axs[1].legend()
axs[1].set_xlim(0, 100000)
axs[1].set_title(r"$\mathrm{std}(e)$")
axs[1].set_xlabel("Time (yr)")

# %%
def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_eccentricity_over_time(ax, steps, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step["inc"][:, 1:], axis=1))
        mean_e.extend(np.mean(step["inc"][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, alpha=0.2, ec="face")

def plot_std_eccentricity_over_time(ax, steps, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step["inc"][:, 1:], axis=1))
    ax.plot(time, np.array(std_e), **kwargs)


fig, axs = plt.subplots(1, 2, figsize=(16, 6))

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        plot_eccentricity_over_time(axs[0], sorted_steps, label=make_label(k))
        plot_std_eccentricity_over_time(axs[1], sorted_steps, label=make_label(k))


axs[0].legend()
axs[0].set_xlim(0, 100000)
axs[0].set_title(r"$\mathrm{mean}(e) \pm \mathrm{std}(e)$")
axs[0].set_xlabel("Time (yr)")

axs[1].legend()
axs[1].set_xlim(0, 100000)
axs[1].set_title(r"$\mathrm{std}(e)$")
axs[1].set_xlabel("Time (yr)")

# %%
def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_eccentricity_over_time(ax, steps, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step["inc"][:, 1:], axis=1))
        mean_e.extend(np.mean(step["inc"][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, alpha=0.2, ec="face")

def plot_std_eccentricity_over_time(ax, steps, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step["inc"][:, 1:], axis=1))
    ax.plot(time, np.array(std_e), **kwargs)


fig, axs = plt.subplots(1, 2, figsize=(16, 6))

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        plot_eccentricity_over_time(axs[0], sorted_steps, label=make_label(k))
        plot_std_eccentricity_over_time(axs[1], sorted_steps, label=make_label(k))


axs[0].legend()
axs[0].set_xlim(0, 100000)
axs[0].set_title(r"$\mathrm{mean}(i) \pm \mathrm{std}(i)$")
axs[0].set_xlabel("Time (yr)")

axs[1].legend()
axs[1].set_xlim(0, 100000)
axs[1].set_title(r"$\mathrm{std}(i)$")
axs[1].set_xlabel("Time (yr)")

# %%
import h5py
import util

column = "a"
mathtext = "a"

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_over_time(ax, steps, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
        mean_e.extend(np.mean(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, alpha=0.2, ec="face")

def plot_std_over_time(ax, steps, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(std_e), **kwargs)


fig, axs = plt.subplots(1, 2, figsize=(16, 6))

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        plot_over_time(axs[0], sorted_steps, label=make_label(k))
        plot_std_over_time(axs[1], sorted_steps, label=make_label(k))


axs[0].legend()
axs[0].set_xlim(0, 100000)
axs[0].set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
axs[0].set_xlabel("Time (yr)")

axs[1].legend()
axs[1].set_xlim(0, 100000)
axs[0].set_title(rf"$\mathrm{{std}}({mathtext})$")
axs[1].set_xlabel("Time (yr)")

# %%
import h5py
import util

column = "a"
mathtext = "a"

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_over_time(ax, steps, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
        mean_e.extend(np.mean(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, alpha=0.2, ec="face")

def plot_std_over_time(ax, steps, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(std_e), **kwargs)


fig, axs = plt.subplots(1, 2, figsize=(16, 6))

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        plot_over_time(axs[0], sorted_steps, label=make_label(k))
        plot_std_over_time(axs[1], sorted_steps, label=make_label(k))


axs[0].legend()
axs[0].set_xlim(0, 100000)
axs[0].set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
axs[0].set_xlabel("Time (yr)")

axs[1].legend()
axs[1].set_xlim(0, 100000)
axs[1].set_title(rf"$\mathrm{{std}}({mathtext})$")
axs[1].set_xlabel("Time (yr)")

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
        mean_e.extend(np.mean(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, alpha=0.2, ec="face")

    ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, column, mathtext, **kwargs):
    plot_over_time(axs[0], column, mathtext)
    plot_std_over_time(axs[1], column, mathtext)


fig, axs = plt.subplots(1, 2, figsize=(16, 6))

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        double_plot(axs, "a", "a")

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
        mean_e.extend(np.mean(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, alpha=0.2, ec="face")

    ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext)
    plot_std_over_time(axs[1], steps, column, mathtext)


fig, axs = plt.subplots(1, 2, figsize=(16, 6))

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        double_plot(axs, steps, "a", "a")

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
        mean_e.extend(np.mean(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, alpha=0.2, ec="face")

    ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext)
    plot_std_over_time(axs[1], steps, column, mathtext)


fig, axs = plt.subplots(1, 2, figsize=(16, 6))

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        double_plot(axs, sorted_steps, "a", "a")

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
        mean_e.extend(np.mean(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, alpha=0.2, ec="face")

    ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext)
    plot_std_over_time(axs[1], steps, column, mathtext)


fig, axs = plt.subplots(1, 2, figsize=(16, 6))

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        double_plot(axs, sorted_steps, "a", "a", label=make_label(k))

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
        mean_e.extend(np.mean(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, alpha=0.2, ec="face")

    ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext, **kwargs)
    plot_std_over_time(axs[1], steps, column, mathtext, **kwargs)


fig, axs = plt.subplots(1, 2, figsize=(16, 6))

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        double_plot(axs, sorted_steps, "a", "a", label=make_label(k))

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
        mean_e.extend(np.mean(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, alpha=0.2, ec="face")

    ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext, **kwargs)
    plot_std_over_time(axs[1], steps, column, mathtext, **kwargs)


fig, axs = plt.subplots(3, 2, figsize=(16, 18))

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        double_plot(axs[0], sorted_steps, "a", "a", label=make_label(k))
        double_plot(axs[1], sorted_steps, "ecc", "e", label=make_label(k))
        double_plot(axs[2], sorted_steps, "inc", "i", label=make_label(k))

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
        mean_e.extend(np.mean(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, alpha=0.2, ec="face")

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext, **kwargs)
    plot_std_over_time(axs[1], steps, column, mathtext, **kwargs)


fig, axs = plt.subplots(3, 2, figsize=(16, 18))

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        double_plot(axs[0], sorted_steps, "a", "a", label=make_label(k))
        double_plot(axs[1], sorted_steps, "ecc", "e", label=make_label(k))
        double_plot(axs[2], sorted_steps, "inc", "i", label=make_label(k))

fig.figlegend()

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
        mean_e.extend(np.mean(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, alpha=0.2, ec="face")

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext, **kwargs)
    plot_std_over_time(axs[1], steps, column, mathtext, **kwargs)


fig, axs = plt.subplots(3, 2, figsize=(16, 18))

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        double_plot(axs[0], sorted_steps, "a", "a", label=make_label(k))
        double_plot(axs[1], sorted_steps, "ecc", "e", label=make_label(k))
        double_plot(axs[2], sorted_steps, "inc", "i", label=make_label(k))

plt.figlegend()

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
        mean_e.extend(np.mean(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, alpha=0.2, ec="face")

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext, **kwargs)
    plot_std_over_time(axs[1], steps, column, mathtext, **kwargs)


fig, axs = plt.subplots(3, 2, figsize=(16, 18))

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        double_plot(axs[0], sorted_steps, "a", "a", label=make_label(k))
        double_plot(axs[1], sorted_steps, "ecc", "e", label=make_label(k))
        double_plot(axs[2], sorted_steps, "inc", "i", label=make_label(k))

plt.figlegend(axs[0].get_lines())

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
        mean_e.extend(np.mean(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, alpha=0.2, ec="face")

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext, **kwargs)
    plot_std_over_time(axs[1], steps, column, mathtext, **kwargs)


fig, axs = plt.subplots(3, 2, figsize=(16, 18))

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        double_plot(axs[0], sorted_steps, "a", "a", label=make_label(k))
        double_plot(axs[1], sorted_steps, "ecc", "e", label=make_label(k))
        double_plot(axs[2], sorted_steps, "inc", "i", label=make_label(k))

plt.figlegend(axs[0][0].get_lines())

# %%
axs[0][0].get_lines()

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
        mean_e.extend(np.mean(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, alpha=0.2, ec="face")

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext, **kwargs)
    plot_std_over_time(axs[1], steps, column, mathtext, **kwargs)


fig, axs = plt.subplots(3, 2, figsize=(16, 18))

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        double_plot(axs[0], sorted_steps, "a", "a", label=make_label(k))
        double_plot(axs[1], sorted_steps, "ecc", "e", label=make_label(k))
        double_plot(axs[2], sorted_steps, "inc", "i", label=make_label(k))

plt.figlegend(handles=axs[0][0].get_lines())

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
        mean_e.extend(np.mean(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, alpha=0.2, ec="face")

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext, **kwargs)
    plot_std_over_time(axs[1], steps, column, mathtext, **kwargs)


fig, axs = plt.subplots(3, 2, figsize=(16, 18))

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        double_plot(axs[0], sorted_steps, "a", "a", label=make_label(k))
        double_plot(axs[1], sorted_steps, "ecc", "e", label=make_label(k))
        double_plot(axs[2], sorted_steps, "inc", "i", label=make_label(k))

fig.legend(handles=axs[0][0].get_lines())

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
        mean_e.extend(np.mean(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, alpha=0.2, ec="face")

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext, **kwargs)
    plot_std_over_time(axs[1], steps, column, mathtext, **kwargs)


fig, axs = plt.subplots(3, 2, figsize=(16, 18))

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        double_plot(axs[0], sorted_steps, "a", "a", label=make_label(k))
        double_plot(axs[1], sorted_steps, "ecc", "e", label=make_label(k))
        double_plot(axs[2], sorted_steps, "inc", "i", label=make_label(k))

fig.legend(handles=axs[0][0].get_lines(), ncol=4)

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
        mean_e.extend(np.mean(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, alpha=0.2, ec="face")

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext, **kwargs)
    plot_std_over_time(axs[1], steps, column, mathtext, **kwargs)


fig, axs = plt.subplots(3, 2, figsize=(16, 18))

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        double_plot(axs[0], sorted_steps, "a", "a", label=make_label(k))
        double_plot(axs[1], sorted_steps, "ecc", "e", label=make_label(k))
        double_plot(axs[2], sorted_steps, "inc", "i", label=make_label(k))

# fig.legend(handles=axs[0][0].get_lines(), ncol=4)

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
        mean_e.extend(np.mean(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, alpha=0.2, ec="face")

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext, **kwargs)
    plot_std_over_time(axs[1], steps, column, mathtext, **kwargs)


fig, axs = plt.subplots(3, 2, figsize=(16, 18))

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        double_plot(axs[0], sorted_steps, "a", "a", label=make_label(k))
        double_plot(axs[1], sorted_steps, "ecc", "e", label=make_label(k))
        double_plot(axs[2], sorted_steps, "inc", "i", label=make_label(k))

fig.legend(handles=axs[0][0].get_lines(), ncol=4, loc="top center")

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
        mean_e.extend(np.mean(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, alpha=0.2, ec="face")

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext, **kwargs)
    plot_std_over_time(axs[1], steps, column, mathtext, **kwargs)


fig, axs = plt.subplots(3, 2, figsize=(16, 18))

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        double_plot(axs[0], sorted_steps, "a", "a", label=make_label(k))
        double_plot(axs[1], sorted_steps, "ecc", "e", label=make_label(k))
        double_plot(axs[2], sorted_steps, "inc", "i", label=make_label(k))

fig.legend(handles=axs[0][0].get_lines(), ncol=4, loc="upper center")

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
        mean_e.extend(np.mean(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, alpha=0.2, ec="face")

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext, **kwargs)
    plot_std_over_time(axs[1], steps, column, mathtext, **kwargs)


fig, axs = plt.subplots(3, 2, figsize=(16, 18), constrained_layout=True)

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        double_plot(axs[0], sorted_steps, "a", "a", label=make_label(k))
        double_plot(axs[1], sorted_steps, "ecc", "e", label=make_label(k))
        double_plot(axs[2], sorted_steps, "inc", "i", label=make_label(k))

fig.legend(handles=axs[0][0].get_lines(), ncol=4, loc="upper center")

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
        mean_e.extend(np.mean(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, alpha=0.2, ec="face")

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext, **kwargs)
    plot_std_over_time(axs[1], steps, column, mathtext, **kwargs)


fig, axs = plt.subplots(3, 2, figsize=(16, 18), constrained_layout=True)

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        double_plot(axs[0], sorted_steps, "a", "a", label=make_label(k))
        double_plot(axs[1], sorted_steps, "ecc", "e", label=make_label(k))
        double_plot(axs[2], sorted_steps, "inc", "i", label=make_label(k))

fig.legend(handles=axs[0][0].get_lines(), ncol=4, loc="upper center", frameon=True)

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
        mean_e.extend(np.mean(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, alpha=0.2, ec="face")

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext, **kwargs)
    plot_std_over_time(axs[1], steps, column, mathtext, **kwargs)


fig, axs = plt.subplots(3, 2, figsize=(16, 18), tight_layout=True)

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        double_plot(axs[0], sorted_steps, "a", "a", label=make_label(k))
        double_plot(axs[1], sorted_steps, "ecc", "e", label=make_label(k))
        double_plot(axs[2], sorted_steps, "inc", "i", label=make_label(k))

fig.legend(handles=axs[0][0].get_lines(), ncol=4, loc="upper center", frameon=True)

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
        mean_e.extend(np.mean(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, alpha=0.2, ec="face")

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext, **kwargs)
    plot_std_over_time(axs[1], steps, column, mathtext, **kwargs)


fig, axs = plt.subplots(3, 2, figsize=(16, 18))

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        double_plot(axs[0], sorted_steps, "a", "a", label=make_label(k))
        double_plot(axs[1], sorted_steps, "ecc", "e", label=make_label(k))
        double_plot(axs[2], sorted_steps, "inc", "i", label=make_label(k))

fig.legend(handles=axs[0][0].get_lines(), ncol=4, loc="upper center", frameon=True)
plt.tight_layout()

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
        mean_e.extend(np.mean(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, alpha=0.2, ec="face")

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext, **kwargs)
    plot_std_over_time(axs[1], steps, column, mathtext, **kwargs)


fig, axs = plt.subplots(3, 2, figsize=(16, 18))

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        double_plot(axs[0], sorted_steps, "a", "a", label=make_label(k))
        double_plot(axs[1], sorted_steps, "ecc", "e", label=make_label(k))
        double_plot(axs[2], sorted_steps, "inc", "i", label=make_label(k))

plt.legend(handles=axs[0][0].get_lines(), ncol=4, loc="upper center", frameon=True)
plt.tight_layout()

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
        mean_e.extend(np.mean(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, alpha=0.2, ec="face")

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext, **kwargs)
    plot_std_over_time(axs[1], steps, column, mathtext, **kwargs)


fig, axs = plt.subplots(3, 2, figsize=(16, 18))

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        double_plot(axs[0], sorted_steps, "a", "a", label=make_label(k))
        double_plot(axs[1], sorted_steps, "ecc", "e", label=make_label(k))
        double_plot(axs[2], sorted_steps, "inc", "i", label=make_label(k))

plt.figlegend(handles=axs[0][0].get_lines(), ncol=4, loc="upper center", frameon=True)
plt.tight_layout()

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
        mean_e.extend(np.mean(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, alpha=0.2, ec="face")

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext, **kwargs)
    plot_std_over_time(axs[1], steps, column, mathtext, **kwargs)


fig, axs = plt.subplots(3, 2, figsize=(16, 18))

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        double_plot(axs[0], sorted_steps, "a", "a", label=make_label(k))
        double_plot(axs[1], sorted_steps, "ecc", "e", label=make_label(k))
        double_plot(axs[2], sorted_steps, "inc", "i", label=make_label(k))

fig.suptitle("Time evolution of orbits (N=100)")

plt.figlegend(handles=axs[0][0].get_lines(), ncol=4, loc="upper center", frameon=True)
plt.tight_layout()

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
        mean_e.extend(np.mean(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, alpha=0.2, ec="face")

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext, **kwargs)
    plot_std_over_time(axs[1], steps, column, mathtext, **kwargs)


fig, axs = plt.subplots(3, 2, figsize=(16, 18), constrained_layout=True)

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        double_plot(axs[0], sorted_steps, "a", "a", label=make_label(k))
        double_plot(axs[1], sorted_steps, "ecc", "e", label=make_label(k))
        double_plot(axs[2], sorted_steps, "inc", "i", label=make_label(k))

fig.suptitle("Time evolution of orbits (N=100)")

plt.figlegend(handles=axs[0][0].get_lines(), ncol=4, loc="upper center", frameon=True)

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
        mean_e.extend(np.mean(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, alpha=0.2, ec="face")

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext, **kwargs)
    plot_std_over_time(axs[1], steps, column, mathtext, **kwargs)


fig, axs = plt.subplots(3, 2, figsize=(8, 9), constrained_layout=True)

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        double_plot(axs[0], sorted_steps, "a", "a", label=make_label(k))
        double_plot(axs[1], sorted_steps, "ecc", "e", label=make_label(k))
        double_plot(axs[2], sorted_steps, "inc", "i", label=make_label(k))

fig.suptitle("Time evolution of orbits (N=100)")

plt.figlegend(handles=axs[0][0].get_lines(), ncol=4, loc="upper center", frameon=True)

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
        mean_e.extend(np.mean(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - std_e, np.array(mean_e) + std_e, alpha=0.2, ec="face")

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext, **kwargs)
    plot_std_over_time(axs[1], steps, column, mathtext, **kwargs)


fig, axs = plt.subplots(3, 2, figsize=(12, 15), constrained_layout=True)

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        double_plot(axs[0], sorted_steps, "a", "a", label=make_label(k))
        double_plot(axs[1], sorted_steps, "ecc", "e", label=make_label(k))
        double_plot(axs[2], sorted_steps, "inc", "i", label=make_label(k))

fig.suptitle("Time evolution of orbits (N=100)")

plt.figlegend(handles=axs[0][0].get_lines(), ncol=4, loc="upper center", frameon=True)

# %%
import matplotlib.pyplot as plt

for k, v in sims.items():
    if v["n_particles"] > 50 and v["drag_coefficient"] == 0:
        runtimes = v["runtimes"].T
        runtimes2 = sims[(1, k[1], k[2])]["runtimes"].T

        Ns = []
        for N in np.arange(runtimes[0, 0], runtimes[0, -1] - 1, -1):
            Ns.append((N - 1, np.median(runtimes[1, runtimes[0] == N]) / np.median(runtimes2[1, runtimes2[0] == N])))
        Ns = np.array(Ns).T
        plt.plot(Ns[0], Ns[1], label=k[1:])

plt.ylabel("Runtime speed ratio")
plt.xlabel("Planets remaining")
plt.legend()
plt.show()

# %%
import matplotlib.pyplot as plt

for k, v in sims.items():
    if v["n_particles"] > 50 and v["drag_coefficient"] == 0:
        runtimes = v["runtimes"].T
        runtimes2 = sims[(1, k[1], k[2])]["runtimes"].T

        Ns = []
        for N in np.arange(runtimes[0, 0], runtimes[0, -1] - 1, -1):
            Ns.append((N - 1, np.median(runtimes[1, runtimes[0] == N]) / np.median(runtimes2[1, runtimes2[0] == N])))
        Ns = np.array(Ns).T
        plt.plot(Ns[0], Ns[1])

plt.ylabel("Runtime speed ratio")
plt.xlabel("Planets remaining")
plt.show()

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
        mean_e.extend(np.mean(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - np.array(std_e) * 2, np.array(mean_e) + np.array(std_e) * 2, alpha=0.2, ec="face")

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext, **kwargs)
    plot_std_over_time(axs[1], steps, column, mathtext, **kwargs)


fig, axs = plt.subplots(3, 2, figsize=(12, 15), constrained_layout=True)

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        double_plot(axs[0], sorted_steps, "a", "a", label=make_label(k))
        double_plot(axs[1], sorted_steps, "ecc", "e", label=make_label(k))
        double_plot(axs[2], sorted_steps, "inc", "i", label=make_label(k))

fig.suptitle("Time evolution of orbits (N=100)")

plt.figlegend(handles=axs[0][0].get_lines(), ncol=4, loc="upper center", frameon=True)

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
        mean_e.extend(np.mean(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - np.array(std_e), np.array(mean_e) + np.array(std_e), alpha=0.2, ec="face")

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext, **kwargs)
    plot_std_over_time(axs[1], steps, column, mathtext, **kwargs)


fig, axs = plt.subplots(3, 2, figsize=(12, 15), constrained_layout=True)

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        double_plot(axs[0], sorted_steps, "a", "a", label=make_label(k))
        double_plot(axs[1], sorted_steps, "ecc", "e", label=make_label(k))
        double_plot(axs[2], sorted_steps, "inc", "i", label=make_label(k))

fig.suptitle("Time evolution of orbits (N=100)")

plt.figlegend(handles=axs[0][0].get_lines(), ncol=4, loc="upper center", frameon=True)

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_over_time(ax, steps, column, mathtext, **kwargs):
    e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        e.extend(step[column][:, 1:])
    e = np.array(e)
    ax.plot(time, np.mean(e, axis=1), **kwargs)
    # ax.fill_between(time, np.array(mean_e) - np.array(std_e), np.array(mean_e) + np.array(std_e), alpha=0.2, ec="face")

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext, **kwargs)
    plot_std_over_time(axs[1], steps, column, mathtext, **kwargs)


fig, axs = plt.subplots(3, 2, figsize=(12, 15), constrained_layout=True)

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        double_plot(axs[0], sorted_steps, "a", "a", label=make_label(k))
        double_plot(axs[1], sorted_steps, "ecc", "e", label=make_label(k))
        double_plot(axs[2], sorted_steps, "inc", "i", label=make_label(k))

fig.suptitle("Time evolution of orbits (N=100)")

plt.figlegend(handles=axs[0][0].get_lines(), ncol=4, loc="upper center", frameon=True)

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_over_time(ax, steps, column, mathtext, **kwargs):
    e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        e.extend(step[column][:, 1:])
        print(step[column][:, 1:])
    e = np.array(e)
    ax.plot(time, np.mean(e, axis=1), **kwargs)
    # ax.fill_between(time, np.array(mean_e) - np.array(std_e), np.array(mean_e) + np.array(std_e), alpha=0.2, ec="face")

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext, **kwargs)
    plot_std_over_time(axs[1], steps, column, mathtext, **kwargs)


fig, axs = plt.subplots(3, 2, figsize=(12, 15), constrained_layout=True)

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        double_plot(axs[0], sorted_steps, "a", "a", label=make_label(k))
        double_plot(axs[1], sorted_steps, "ecc", "e", label=make_label(k))
        double_plot(axs[2], sorted_steps, "inc", "i", label=make_label(k))

fig.suptitle("Time evolution of orbits (N=100)")

plt.figlegend(handles=axs[0][0].get_lines(), ncol=4, loc="upper center", frameon=True)

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_over_time(ax, steps, column, mathtext, **kwargs):
    e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        e.extend(step[column][:, 1:])
        print(step[column][:, 1:].shape)
    e = np.array(e)
    ax.plot(time, np.mean(e, axis=1), **kwargs)
    # ax.fill_between(time, np.array(mean_e) - np.array(std_e), np.array(mean_e) + np.array(std_e), alpha=0.2, ec="face")

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext, **kwargs)
    plot_std_over_time(axs[1], steps, column, mathtext, **kwargs)


fig, axs = plt.subplots(3, 2, figsize=(12, 15), constrained_layout=True)

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        double_plot(axs[0], sorted_steps, "a", "a", label=make_label(k))
        double_plot(axs[1], sorted_steps, "ecc", "e", label=make_label(k))
        double_plot(axs[2], sorted_steps, "inc", "i", label=make_label(k))

fig.suptitle("Time evolution of orbits (N=100)")

plt.figlegend(handles=axs[0][0].get_lines(), ncol=4, loc="upper center", frameon=True)

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_over_time(ax, steps, column, mathtext, **kwargs):
    e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        e.extend(step[column][:, 1:])
    print(e)
    ax.plot(time, np.mean(e, axis=1), **kwargs)
    # ax.fill_between(time, np.array(mean_e) - np.array(std_e), np.array(mean_e) + np.array(std_e), alpha=0.2, ec="face")

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext, **kwargs)
    plot_std_over_time(axs[1], steps, column, mathtext, **kwargs)


fig, axs = plt.subplots(3, 2, figsize=(12, 15), constrained_layout=True)

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        double_plot(axs[0], sorted_steps, "a", "a", label=make_label(k))
        double_plot(axs[1], sorted_steps, "ecc", "e", label=make_label(k))
        double_plot(axs[2], sorted_steps, "inc", "i", label=make_label(k))

fig.suptitle("Time evolution of orbits (N=100)")

plt.figlegend(handles=axs[0][0].get_lines(), ncol=4, loc="upper center", frameon=True)

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_over_time(ax, steps, column, mathtext, **kwargs):
    e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        e.extend(step[column][:, 1:])
    e = np.row_stack(e)
    ax.plot(time, np.mean(e, axis=1), **kwargs)
    # ax.fill_between(time, np.array(mean_e) - np.array(std_e), np.array(mean_e) + np.array(std_e), alpha=0.2, ec="face")

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        std_e.extend(np.std(step[column][:, 1:], axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext, **kwargs)
    plot_std_over_time(axs[1], steps, column, mathtext, **kwargs)


fig, axs = plt.subplots(3, 2, figsize=(12, 15), constrained_layout=True)

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        double_plot(axs[0], sorted_steps, "a", "a", label=make_label(k))
        double_plot(axs[1], sorted_steps, "ecc", "e", label=make_label(k))
        double_plot(axs[2], sorted_steps, "inc", "i", label=make_label(k))

fig.suptitle("Time evolution of orbits (N=100)")

plt.figlegend(handles=axs[0][0].get_lines(), ncol=4, loc="upper center", frameon=True)

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        data = step[column][:, 1:]
        std_e.extend(np.std(data, axis=1))
        mean_e.extend(np.mean(data, axis=1))
        ax.scatter(t, data)

    ax.plot(time, np.array(mean_e), **kwargs)
    # ax.fill_between(time, np.array(mean_e) - np.array(std_e), np.array(mean_e) + np.array(std_e), alpha=0.2, ec="face")

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        data = step[column][:, 1:]
        std_e.extend(np.std(data, axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext, **kwargs)
    plot_std_over_time(axs[1], steps, column, mathtext, **kwargs)


fig, axs = plt.subplots(3, 2, figsize=(12, 15), constrained_layout=True)

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        double_plot(axs[0], sorted_steps, "a", "a", label=make_label(k))
        double_plot(axs[1], sorted_steps, "ecc", "e", label=make_label(k))
        double_plot(axs[2], sorted_steps, "inc", "i", label=make_label(k))

fig.suptitle("Time evolution of orbits (N=100)")

plt.figlegend(handles=axs[0][0].get_lines(), ncol=4, loc="upper center", frameon=True)

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        data = step[column][:, 1:]
        std_e.extend(np.std(data, axis=1))
        mean_e.extend(np.mean(data, axis=1))
        ax.scatter(time, data)

    ax.plot(time, np.array(mean_e), **kwargs)
    # ax.fill_between(time, np.array(mean_e) - np.array(std_e), np.array(mean_e) + np.array(std_e), alpha=0.2, ec="face")

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        data = step[column][:, 1:]
        std_e.extend(np.std(data, axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext, **kwargs)
    plot_std_over_time(axs[1], steps, column, mathtext, **kwargs)


fig, axs = plt.subplots(3, 2, figsize=(12, 15), constrained_layout=True)

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        double_plot(axs[0], sorted_steps, "a", "a", label=make_label(k))
        double_plot(axs[1], sorted_steps, "ecc", "e", label=make_label(k))
        double_plot(axs[2], sorted_steps, "inc", "i", label=make_label(k))

fig.suptitle("Time evolution of orbits (N=100)")

plt.figlegend(handles=axs[0][0].get_lines(), ncol=4, loc="upper center", frameon=True)

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        data = step[column][:, 1:]
        std_e.extend(np.std(data, axis=1))
        mean_e.extend(np.mean(data, axis=1))
        ax.scatter(step["time"], data)

    ax.plot(time, np.array(mean_e), **kwargs)
    # ax.fill_between(time, np.array(mean_e) - np.array(std_e), np.array(mean_e) + np.array(std_e), alpha=0.2, ec="face")

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        data = step[column][:, 1:]
        std_e.extend(np.std(data, axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext, **kwargs)
    plot_std_over_time(axs[1], steps, column, mathtext, **kwargs)


fig, axs = plt.subplots(3, 2, figsize=(12, 15), constrained_layout=True)

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        double_plot(axs[0], sorted_steps, "a", "a", label=make_label(k))
        double_plot(axs[1], sorted_steps, "ecc", "e", label=make_label(k))
        double_plot(axs[2], sorted_steps, "inc", "i", label=make_label(k))

fig.suptitle("Time evolution of orbits (N=100)")

plt.figlegend(handles=axs[0][0].get_lines(), ncol=4, loc="upper center", frameon=True)

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        data = step[column][:, 1:]
        std_e.extend(np.std(data, axis=1))
        mean_e.extend(np.mean(data, axis=1))

    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - np.array(std_e), np.array(mean_e) + np.array(std_e), alpha=0.2, ec="face")

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        data = step[column][:, 1:]
        std_e.extend(np.std(data, axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext, **kwargs)
    plot_std_over_time(axs[1], steps, column, mathtext, **kwargs)


fig, axs = plt.subplots(3, 2, figsize=(12, 15), constrained_layout=True)

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        double_plot(axs[0], sorted_steps, "a", "a", label=make_label(k))
        double_plot(axs[1], sorted_steps, "ecc", "e", label=make_label(k))
        double_plot(axs[2], sorted_steps, "inc", "i", label=make_label(k))

fig.suptitle("Time evolution of orbits (N=100)")

plt.figlegend(handles=axs[0][0].get_lines(), ncol=4, loc="upper center", frameon=True)

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    mean_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        data = step[column][:, 1:]
        std_e.extend(np.std(data, axis=1))
        mean_e.extend(np.mean(data, axis=1))

    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - np.array(std_e), np.array(mean_e) + np.array(std_e), alpha=0.1, ec="face")

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        data = step[column][:, 1:]
        std_e.extend(np.std(data, axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext, **kwargs)
    plot_std_over_time(axs[1], steps, column, mathtext, **kwargs)


fig, axs = plt.subplots(3, 2, figsize=(12, 15), constrained_layout=True)

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        double_plot(axs[0], sorted_steps, "a", "a", label=make_label(k))
        double_plot(axs[1], sorted_steps, "ecc", "e", label=make_label(k))
        double_plot(axs[2], sorted_steps, "inc", "i", label=make_label(k))

fig.suptitle("Time evolution of orbits (N=100)")

plt.figlegend(handles=axs[0][0].get_lines(), ncol=4, loc="upper center", frameon=True)

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    mean_e = []
    min_e = []
    max_e = []

    time = []
    for step in steps:
        time.extend(step["time"])
        data = step[column][:, 1:]
        std_e.extend(np.std(data, axis=1))
        mean_e.extend(np.mean(data, axis=1))
        min_e.extend(np.min(data, axis=1))
        max_e.extend(np.max(data, axis=1))

    ax.plot(time, np.array(mean_e), **kwargs)
    # ax.fill_between(time, np.array(mean_e) - np.array(std_e), np.array(mean_e) + np.array(std_e), alpha=0.1, ec="face")
    ax.fill_between(time, np.array(min_e), np.array(max_e), alpha=0.1, ec="face")

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        data = step[column][:, 1:]
        std_e.extend(np.std(data, axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext, **kwargs)
    plot_std_over_time(axs[1], steps, column, mathtext, **kwargs)


fig, axs = plt.subplots(3, 2, figsize=(12, 15), constrained_layout=True)

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        double_plot(axs[0], sorted_steps, "a", "a", label=make_label(k))
        double_plot(axs[1], sorted_steps, "ecc", "e", label=make_label(k))
        double_plot(axs[2], sorted_steps, "inc", "i", label=make_label(k))

fig.suptitle("Time evolution of orbits (N=100)")

plt.figlegend(handles=axs[0][0].get_lines(), ncol=4, loc="upper center", frameon=True)

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    mean_e = []
    min_e = []
    max_e = []

    time = []
    for step in steps:
        time.extend(step["time"])
        data = step[column][:, 1:]
        std_e.extend(np.std(data, axis=1))
        mean_e.extend(np.mean(data, axis=1))
        min_e.extend(np.percentile(data, 0.25, axis=1))
        max_e.extend(np.percentile(data, 0.75, axis=1))

    ax.plot(time, np.array(mean_e), **kwargs)
    # ax.fill_between(time, np.array(mean_e) - np.array(std_e), np.array(mean_e) + np.array(std_e), alpha=0.1, ec="face")
    ax.fill_between(time, np.array(min_e), np.array(max_e), alpha=0.1, ec="face")

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        data = step[column][:, 1:]
        std_e.extend(np.std(data, axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext, **kwargs)
    plot_std_over_time(axs[1], steps, column, mathtext, **kwargs)


fig, axs = plt.subplots(3, 2, figsize=(12, 15), constrained_layout=True)

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        double_plot(axs[0], sorted_steps, "a", "a", label=make_label(k))
        double_plot(axs[1], sorted_steps, "ecc", "e", label=make_label(k))
        double_plot(axs[2], sorted_steps, "inc", "i", label=make_label(k))

fig.suptitle("Time evolution of orbits (N=100)")

plt.figlegend(handles=axs[0][0].get_lines(), ncol=4, loc="upper center", frameon=True)

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    mean_e = []
    min_e = []
    max_e = []

    time = []
    for step in steps:
        time.extend(step["time"])
        data = step[column][:, 1:]
        std_e.extend(np.std(data, axis=1))
        mean_e.extend(np.mean(data, axis=1))
        min_e.extend(np.percentile(data, 25, axis=1))
        max_e.extend(np.percentile(data, 75, axis=1))

    ax.plot(time, np.array(mean_e), **kwargs)
    # ax.fill_between(time, np.array(mean_e) - np.array(std_e), np.array(mean_e) + np.array(std_e), alpha=0.1, ec="face")
    ax.fill_between(time, np.array(min_e), np.array(max_e), alpha=0.1, ec="face")

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        data = step[column][:, 1:]
        std_e.extend(np.std(data, axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext, **kwargs)
    plot_std_over_time(axs[1], steps, column, mathtext, **kwargs)


fig, axs = plt.subplots(3, 2, figsize=(12, 15), constrained_layout=True)

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        double_plot(axs[0], sorted_steps, "a", "a", label=make_label(k))
        double_plot(axs[1], sorted_steps, "ecc", "e", label=make_label(k))
        double_plot(axs[2], sorted_steps, "inc", "i", label=make_label(k))

fig.suptitle("Time evolution of orbits (N=100)")

plt.figlegend(handles=axs[0][0].get_lines(), ncol=4, loc="upper center", frameon=True)

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    mean_e = []
    min_e = []
    max_e = []

    time = []
    for step in steps:
        time.extend(step["time"])
        data = step[column][:, 1:]
        std_e.extend(np.std(data, axis=1))
        mean_e.extend(np.mean(data, axis=1))
        min_e.extend(np.percentile(data, 25, axis=1))
        max_e.extend(np.percentile(data, 75, axis=1))

    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - np.array(std_e), np.array(mean_e) + np.array(std_e), alpha=0.1, ec="face")
    # ax.fill_between(time, np.array(min_e), np.array(max_e), alpha=0.1, ec="face")

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        data = step[column][:, 1:]
        std_e.extend(np.std(data, axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext, **kwargs)
    plot_std_over_time(axs[1], steps, column, mathtext, **kwargs)


fig, axs = plt.subplots(3, 2, figsize=(12, 15), constrained_layout=True)

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        double_plot(axs[0], sorted_steps, "a", "a", label=make_label(k))
        double_plot(axs[1], sorted_steps, "ecc", "e", label=make_label(k))
        double_plot(axs[2], sorted_steps, "inc", "i", label=make_label(k))

fig.suptitle("Time evolution of orbits (N=100)")

plt.figlegend(handles=axs[0][0].get_lines(), ncol=4, loc="upper center", frameon=True)

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    mean_e = []
    min_e = []
    max_e = []

    time = []
    for step in steps:
        time.extend(step["time"])
        data = step[column][:, 1:]
        std_e.extend(np.std(data, axis=1))
        mean_e.extend(np.mean(data, axis=1))
        min_e.extend(np.percentile(data, 25, axis=1))
        max_e.extend(np.percentile(data, 75, axis=1))

    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - np.array(std_e), np.array(mean_e) + np.array(std_e), alpha=0.1, ec="face")
    # ax.fill_between(time, np.array(min_e), np.array(max_e), alpha=0.1, ec="face")

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")
    ax.set_xscale("log")

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        data = step[column][:, 1:]
        std_e.extend(np.std(data, axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")
    ax.set_xscale("log")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext, **kwargs)
    plot_std_over_time(axs[1], steps, column, mathtext, **kwargs)


fig, axs = plt.subplots(3, 2, figsize=(12, 15), constrained_layout=True)

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        double_plot(axs[0], sorted_steps, "a", "a", label=make_label(k))
        double_plot(axs[1], sorted_steps, "ecc", "e", label=make_label(k))
        double_plot(axs[2], sorted_steps, "inc", "i", label=make_label(k))

fig.suptitle("Time evolution of orbits (N=100)")

plt.figlegend(handles=axs[0][0].get_lines(), ncol=4, loc="upper center", frameon=True)

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    mean_e = []
    min_e = []
    max_e = []

    time = []
    for step in steps:
        time.extend(step["time"])
        data = step[column][:, 1:]
        std_e.extend(np.std(data, axis=1))
        mean_e.extend(np.mean(data, axis=1))
        min_e.extend(np.percentile(data, 25, axis=1))
        max_e.extend(np.percentile(data, 75, axis=1))

    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - np.array(std_e), np.array(mean_e) + np.array(std_e), alpha=0.1, ec="face")
    # ax.fill_between(time, np.array(min_e), np.array(max_e), alpha=0.1, ec="face")

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        data = step[column][:, 1:]
        std_e.extend(np.std(data, axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")
    ax.set_xscale("log")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext, **kwargs)
    plot_std_over_time(axs[1], steps, column, mathtext, **kwargs)


fig, axs = plt.subplots(3, 2, figsize=(12, 15), constrained_layout=True)

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        double_plot(axs[0], sorted_steps, "a", "a", label=make_label(k))
        double_plot(axs[1], sorted_steps, "ecc", "e", label=make_label(k))
        double_plot(axs[2], sorted_steps, "inc", "i", label=make_label(k))

fig.suptitle("Time evolution of orbits (N=100)")

plt.figlegend(handles=axs[0][0].get_lines(), ncol=4, loc="upper center", frameon=True)

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    mean_e = []
    min_e = []
    max_e = []

    time = []
    for step in steps:
        time.extend(step["time"])
        data = step[column][:, 1:]
        std_e.extend(np.std(data, axis=1))
        mean_e.extend(np.mean(data, axis=1))
        min_e.extend(np.percentile(data, 25, axis=1))
        max_e.extend(np.percentile(data, 75, axis=1))

    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - np.array(std_e), np.array(mean_e) + np.array(std_e), alpha=0.1, ec="face")
    # ax.fill_between(time, np.array(min_e), np.array(max_e), alpha=0.1, ec="face")

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")
    

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        data = step[column][:, 1:]
        std_e.extend(np.std(data, axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext, **kwargs)
    plot_std_over_time(axs[1], steps, column, mathtext, **kwargs)


fig, axs = plt.subplots(3, 2, figsize=(12, 15), constrained_layout=True)

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        double_plot(axs[0], sorted_steps, "a", "a", label=make_label(k))
        double_plot(axs[1], sorted_steps, "ecc", "e", label=make_label(k))
        double_plot(axs[2], sorted_steps, "inc", "i", label=make_label(k))

fig.suptitle("Time evolution of orbits (N=100)")

plt.figlegend(handles=axs[0][0].get_lines(), ncol=4, loc="upper center", frameon=True)

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    mean_e = []
    # min_e = []
    # max_e = []

    time = []
    for step in steps:
        time.extend(step["time"])
        data = step[column][:, 1:]
        std_e.extend(np.nanstd(data, axis=1))
        mean_e.extend(np.nanmean(data, axis=1))
        # min_e.extend(np.percentile(data, 25, axis=1))
        # max_e.extend(np.percentile(data, 75, axis=1))

    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - np.array(std_e), np.array(mean_e) + np.array(std_e), alpha=0.1, ec="face")
    # ax.fill_between(time, np.array(min_e), np.array(max_e), alpha=0.1, ec="face")

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")
    

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        data = step[column][:, 1:]
        std_e.extend(np.nanstd(data, axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext, **kwargs)
    plot_std_over_time(axs[1], steps, column, mathtext, **kwargs)


fig, axs = plt.subplots(3, 2, figsize=(12, 15), constrained_layout=True)

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        double_plot(axs[0], sorted_steps, "a", "a", label=make_label(k))
        double_plot(axs[1], sorted_steps, "ecc", "e", label=make_label(k))
        double_plot(axs[2], sorted_steps, "inc", "i", label=make_label(k))

fig.suptitle("Time evolution of orbits (N=100)")

plt.figlegend(handles=axs[0][0].get_lines(), ncol=4, loc="upper center", frameon=True)

# %%
sorted_steps

# %%
sorted_steps[0].keys()

# %%
sorted_steps[0]["ptype"]

# %%
sorted_steps[0]["ptype"][0]

# %%
sorted_steps[0]["energy"]

# %%
sorted_steps[0]["energy"][0]

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_energy_over_time(ax, steps, **kwargs):
    E = []
    time = []
    for step in steps:
        time.extend(step["time"])
        E.extend(step["E"])

    ax.plot(time, np.array(E), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$E$")
    ax.set_xlabel("Time (yr)")


def plot_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    mean_e = []
    # min_e = []
    # max_e = []

    time = []
    for step in steps:
        time.extend(step["time"])
        data = step[column][:, 1:]
        std_e.extend(np.nanstd(data, axis=1))
        mean_e.extend(np.nanmean(data, axis=1))
        # min_e.extend(np.percentile(data, 25, axis=1))
        # max_e.extend(np.percentile(data, 75, axis=1))

    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - np.array(std_e), np.array(mean_e) + np.array(std_e), alpha=0.1, ec="face")
    # ax.fill_between(time, np.array(min_e), np.array(max_e), alpha=0.1, ec="face")

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")
    

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        data = step[column][:, 1:]
        std_e.extend(np.nanstd(data, axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext, **kwargs)
    plot_std_over_time(axs[1], steps, column, mathtext, **kwargs)


fig, axs = plt.subplots(4, 2, figsize=(12, 20), constrained_layout=True)

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        plot_energy_over_time(axs[0][1], sorted_steps, label=make_label(k))
        double_plot(axs[1], sorted_steps, "a", "a", label=make_label(k))
        double_plot(axs[2], sorted_steps, "ecc", "e", label=make_label(k))
        double_plot(axs[3], sorted_steps, "inc", "i", label=make_label(k))

fig.suptitle("Time evolution of orbits (N=100)")

plt.figlegend(handles=axs[0][0].get_lines(), ncol=4, loc="upper center", frameon=True)

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_energy_over_time(ax, steps, **kwargs):
    E = []
    time = []
    for step in steps:
        time.extend(step["time"])
        print(step.keys())
        E.extend(step["E"])

    ax.plot(time, np.array(E), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$E$")
    ax.set_xlabel("Time (yr)")


def plot_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    mean_e = []
    # min_e = []
    # max_e = []

    time = []
    for step in steps:
        time.extend(step["time"])
        data = step[column][:, 1:]
        std_e.extend(np.nanstd(data, axis=1))
        mean_e.extend(np.nanmean(data, axis=1))
        # min_e.extend(np.percentile(data, 25, axis=1))
        # max_e.extend(np.percentile(data, 75, axis=1))

    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - np.array(std_e), np.array(mean_e) + np.array(std_e), alpha=0.1, ec="face")
    # ax.fill_between(time, np.array(min_e), np.array(max_e), alpha=0.1, ec="face")

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")
    

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        data = step[column][:, 1:]
        std_e.extend(np.nanstd(data, axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext, **kwargs)
    plot_std_over_time(axs[1], steps, column, mathtext, **kwargs)


fig, axs = plt.subplots(4, 2, figsize=(12, 20), constrained_layout=True)

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        plot_energy_over_time(axs[0][1], sorted_steps, label=make_label(k))
        double_plot(axs[1], sorted_steps, "a", "a", label=make_label(k))
        double_plot(axs[2], sorted_steps, "ecc", "e", label=make_label(k))
        double_plot(axs[3], sorted_steps, "inc", "i", label=make_label(k))

fig.suptitle("Time evolution of orbits (N=100)")

plt.figlegend(handles=axs[0][0].get_lines(), ncol=4, loc="upper center", frameon=True)

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_energy_over_time(ax, steps, **kwargs):
    E = []
    time = []
    for step in steps:
        time.extend(step["time"])
        E.extend(step["energy"])

    ax.plot(time, np.array(E), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$E$")
    ax.set_xlabel("Time (yr)")


def plot_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    mean_e = []
    # min_e = []
    # max_e = []

    time = []
    for step in steps:
        time.extend(step["time"])
        data = step[column][:, 1:]
        std_e.extend(np.nanstd(data, axis=1))
        mean_e.extend(np.nanmean(data, axis=1))
        # min_e.extend(np.percentile(data, 25, axis=1))
        # max_e.extend(np.percentile(data, 75, axis=1))

    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - np.array(std_e), np.array(mean_e) + np.array(std_e), alpha=0.1, ec="face")
    # ax.fill_between(time, np.array(min_e), np.array(max_e), alpha=0.1, ec="face")

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")
    

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        data = step[column][:, 1:]
        std_e.extend(np.nanstd(data, axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext, **kwargs)
    plot_std_over_time(axs[1], steps, column, mathtext, **kwargs)


fig, axs = plt.subplots(4, 2, figsize=(12, 20), constrained_layout=True)

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        plot_energy_over_time(axs[0][1], sorted_steps, label=make_label(k))
        double_plot(axs[1], sorted_steps, "a", "a", label=make_label(k))
        double_plot(axs[2], sorted_steps, "ecc", "e", label=make_label(k))
        double_plot(axs[3], sorted_steps, "inc", "i", label=make_label(k))

fig.suptitle("Time evolution of orbits (N=100)")

plt.figlegend(handles=axs[0][0].get_lines(), ncol=4, loc="upper center", frameon=True)

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_energy_over_time(ax, steps, **kwargs):
    E = []
    time = []
    for step in steps:
        time.extend(step["time"])
        E.extend(step["energy"])

    ax.plot(time, np.array(E), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$E$")
    ax.set_xlabel("Time (yr)")


def plot_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    mean_e = []
    # min_e = []
    # max_e = []

    time = []
    for step in steps:
        time.extend(step["time"])
        data = step[column][:, 1:]
        std_e.extend(np.nanstd(data, axis=1))
        mean_e.extend(np.nanmean(data, axis=1))
        # min_e.extend(np.percentile(data, 25, axis=1))
        # max_e.extend(np.percentile(data, 75, axis=1))

    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - np.array(std_e), np.array(mean_e) + np.array(std_e), alpha=0.1, ec="face")
    # ax.fill_between(time, np.array(min_e), np.array(max_e), alpha=0.1, ec="face")

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")
    

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        data = step[column][:, 1:]
        std_e.extend(np.nanstd(data, axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext, **kwargs)
    plot_std_over_time(axs[1], steps, column, mathtext, **kwargs)


fig, axs = plt.subplots(4, 2, figsize=(12, 20), constrained_layout=True)

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        plot_energy_over_time(axs[0][1], sorted_steps, label=make_label(k))
        double_plot(axs[1], sorted_steps, "a", "a", label=make_label(k))
        double_plot(axs[2], sorted_steps, "ecc", "e", label=make_label(k))
        double_plot(axs[3], sorted_steps, "inc", "i", label=make_label(k))

fig.suptitle("Time evolution of orbits (N=100)")

plt.figlegend(handles=axs[1][0].get_lines(), ncol=4, loc="upper center", frameon=True)

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_energy_over_time(ax, steps, **kwargs):
    E = []
    time = []
    for step in steps:
        time.extend(step["time"])
        E.extend(step["energy"])

    ax.plot(time, np.array(E), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$E$")
    ax.set_xlabel("Time (yr)")

    ax.set_yticks([])


def plot_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    mean_e = []
    # min_e = []
    # max_e = []

    time = []
    for step in steps:
        time.extend(step["time"])
        data = step[column][:, 1:]
        std_e.extend(np.nanstd(data, axis=1))
        mean_e.extend(np.nanmean(data, axis=1))
        # min_e.extend(np.percentile(data, 25, axis=1))
        # max_e.extend(np.percentile(data, 75, axis=1))

    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - np.array(std_e), np.array(mean_e) + np.array(std_e), alpha=0.1, ec="face")
    # ax.fill_between(time, np.array(min_e), np.array(max_e), alpha=0.1, ec="face")

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")
    

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        data = step[column][:, 1:]
        std_e.extend(np.nanstd(data, axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext, **kwargs)
    plot_std_over_time(axs[1], steps, column, mathtext, **kwargs)


fig, axs = plt.subplots(4, 2, figsize=(12, 20), constrained_layout=True)

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        plot_energy_over_time(axs[0][1], sorted_steps, label=make_label(k))
        double_plot(axs[1], sorted_steps, "a", "a", label=make_label(k))
        double_plot(axs[2], sorted_steps, "ecc", "e", label=make_label(k))
        double_plot(axs[3], sorted_steps, "inc", "i", label=make_label(k))

fig.suptitle("Time evolution of orbits (N=100)")

plt.figlegend(handles=axs[1][0].get_lines(), ncol=4, loc="upper center", frameon=True)

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_energy_over_time(ax, steps, **kwargs):
    E = []
    time = []
    for step in steps:
        time.extend(step["time"])
        E.extend(step["energy"])

    ax.plot(time, np.array(E), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$E$")
    ax.set_xlabel("Time (yr)")

    ax.set_yticklabels([])


def plot_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    mean_e = []
    # min_e = []
    # max_e = []

    time = []
    for step in steps:
        time.extend(step["time"])
        data = step[column][:, 1:]
        std_e.extend(np.nanstd(data, axis=1))
        mean_e.extend(np.nanmean(data, axis=1))
        # min_e.extend(np.percentile(data, 25, axis=1))
        # max_e.extend(np.percentile(data, 75, axis=1))

    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - np.array(std_e), np.array(mean_e) + np.array(std_e), alpha=0.1, ec="face")
    # ax.fill_between(time, np.array(min_e), np.array(max_e), alpha=0.1, ec="face")

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")
    

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        data = step[column][:, 1:]
        std_e.extend(np.nanstd(data, axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext, **kwargs)
    plot_std_over_time(axs[1], steps, column, mathtext, **kwargs)


fig, axs = plt.subplots(4, 2, figsize=(12, 20), constrained_layout=True)

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        plot_energy_over_time(axs[0][1], sorted_steps, label=make_label(k))
        double_plot(axs[1], sorted_steps, "a", "a", label=make_label(k))
        double_plot(axs[2], sorted_steps, "ecc", "e", label=make_label(k))
        double_plot(axs[3], sorted_steps, "inc", "i", label=make_label(k))

fig.suptitle("Time evolution of orbits (N=100)")

plt.figlegend(handles=axs[1][0].get_lines(), ncol=4, loc="upper center", frameon=True)

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_remaining_over_time(ax, steps, **kwargs):
    remaining = []
    time = []
    for step in steps:
        time.extend(step["time"])
        remaining.extend(np.sum(~np.isnan(step["mass"]), axis=1))

    ax.plot(time, np.array(remaining), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"Remaining objects")
    ax.set_xlabel("Time (yr)")

def plot_energy_over_time(ax, steps, **kwargs):
    E = []
    time = []
    for step in steps:
        time.extend(step["time"])
        E.extend(step["energy"])

    ax.plot(time, np.array(E), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$E$")
    ax.set_xlabel("Time (yr)")

    ax.set_yticklabels([])


def plot_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    mean_e = []
    # min_e = []
    # max_e = []

    time = []
    for step in steps:
        time.extend(step["time"])
        data = step[column][:, 1:]
        std_e.extend(np.nanstd(data, axis=1))
        mean_e.extend(np.nanmean(data, axis=1))
        # min_e.extend(np.percentile(data, 25, axis=1))
        # max_e.extend(np.percentile(data, 75, axis=1))

    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - np.array(std_e), np.array(mean_e) + np.array(std_e), alpha=0.1, ec="face")
    # ax.fill_between(time, np.array(min_e), np.array(max_e), alpha=0.1, ec="face")

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")
    

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        data = step[column][:, 1:]
        std_e.extend(np.nanstd(data, axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext, **kwargs)
    plot_std_over_time(axs[1], steps, column, mathtext, **kwargs)


fig, axs = plt.subplots(4, 2, figsize=(12, 20), constrained_layout=True)

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        plot_energy_over_time(axs[0][1], sorted_steps, label=make_label(k))
        double_plot(axs[1], sorted_steps, "a", "a", label=make_label(k))
        double_plot(axs[2], sorted_steps, "ecc", "e", label=make_label(k))
        double_plot(axs[3], sorted_steps, "inc", "i", label=make_label(k))

fig.suptitle("Time evolution of orbits (N=100)")

plt.figlegend(handles=axs[1][0].get_lines(), ncol=4, loc="upper center", frameon=True)

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_remaining_over_time(ax, steps, **kwargs):
    remaining = []
    time = []
    for step in steps:
        time.extend(step["time"])
        remaining.extend(np.sum(~np.isnan(step["mass"]), axis=1))

    ax.plot(time, np.array(remaining), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"Remaining objects")
    ax.set_xlabel("Time (yr)")

def plot_energy_over_time(ax, steps, **kwargs):
    E = []
    time = []
    for step in steps:
        time.extend(step["time"])
        E.extend(step["energy"])

    ax.plot(time, np.array(E), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$E$")
    ax.set_xlabel("Time (yr)")

    ax.set_yticklabels([])


def plot_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    mean_e = []
    # min_e = []
    # max_e = []

    time = []
    for step in steps:
        time.extend(step["time"])
        data = step[column][:, 1:]
        std_e.extend(np.nanstd(data, axis=1))
        mean_e.extend(np.nanmean(data, axis=1))
        # min_e.extend(np.percentile(data, 25, axis=1))
        # max_e.extend(np.percentile(data, 75, axis=1))

    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - np.array(std_e), np.array(mean_e) + np.array(std_e), alpha=0.1, ec="face")
    # ax.fill_between(time, np.array(min_e), np.array(max_e), alpha=0.1, ec="face")

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")
    

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        data = step[column][:, 1:]
        std_e.extend(np.nanstd(data, axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext, **kwargs)
    plot_std_over_time(axs[1], steps, column, mathtext, **kwargs)


fig, axs = plt.subplots(4, 2, figsize=(12, 20), constrained_layout=True)

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        plot_remaining_over_time(axs[0][0], sorted_steps, label=make_label(k))
        plot_energy_over_time(axs[0][1], sorted_steps, label=make_label(k))
        double_plot(axs[1], sorted_steps, "a", "a", label=make_label(k))
        double_plot(axs[2], sorted_steps, "ecc", "e", label=make_label(k))
        double_plot(axs[3], sorted_steps, "inc", "i", label=make_label(k))

fig.suptitle("Time evolution of orbits (N=100)")

plt.figlegend(handles=axs[1][0].get_lines(), ncol=4, loc="upper center", frameon=True)

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_remaining_over_time(ax, steps, **kwargs):
    remaining = []
    time = []
    for step in steps:
        time.extend(step["time"])
        remaining.extend(np.sum(~np.isnan(step["mass"]), axis=1))

    ax.plot(time, np.array(remaining), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"Remaining objects")
    ax.set_xlabel("Time (yr)")

    ax.set_ylim(-5, 105)

def plot_energy_over_time(ax, steps, **kwargs):
    E = []
    time = []
    for step in steps:
        time.extend(step["time"])
        E.extend(step["energy"])

    ax.plot(time, np.array(E), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$E$")
    ax.set_xlabel("Time (yr)")

    ax.set_yticklabels([])


def plot_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    mean_e = []
    # min_e = []
    # max_e = []

    time = []
    for step in steps:
        time.extend(step["time"])
        data = step[column][:, 1:]
        std_e.extend(np.nanstd(data, axis=1))
        mean_e.extend(np.nanmean(data, axis=1))
        # min_e.extend(np.percentile(data, 25, axis=1))
        # max_e.extend(np.percentile(data, 75, axis=1))

    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - np.array(std_e), np.array(mean_e) + np.array(std_e), alpha=0.1, ec="face")
    # ax.fill_between(time, np.array(min_e), np.array(max_e), alpha=0.1, ec="face")

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")
    

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        data = step[column][:, 1:]
        std_e.extend(np.nanstd(data, axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext, **kwargs)
    plot_std_over_time(axs[1], steps, column, mathtext, **kwargs)


fig, axs = plt.subplots(4, 2, figsize=(12, 20), constrained_layout=True)

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        plot_remaining_over_time(axs[0][0], sorted_steps, label=make_label(k))
        plot_energy_over_time(axs[0][1], sorted_steps, label=make_label(k))
        double_plot(axs[1], sorted_steps, "a", "a", label=make_label(k))
        double_plot(axs[2], sorted_steps, "ecc", "e", label=make_label(k))
        double_plot(axs[3], sorted_steps, "inc", "i", label=make_label(k))

fig.suptitle("Time evolution of orbits (N=100)")

plt.figlegend(handles=axs[1][0].get_lines(), ncol=4, loc="upper center", frameon=True)

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_remaining_over_time(ax, steps, **kwargs):
    remaining = []
    time = []
    for step in steps:
        time.extend(step["time"])
        remaining.extend(np.sum(~np.isnan(step["mass"]), axis=1) - 1)

    ax.plot(time, np.array(remaining), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"Remaining objects")
    ax.set_xlabel("Time (yr)")

    ax.set_ylim(0, 105)

def plot_energy_over_time(ax, steps, **kwargs):
    E = []
    time = []
    for step in steps:
        time.extend(step["time"])
        E.extend(step["energy"])

    ax.plot(time, np.array(E), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$E$")
    ax.set_xlabel("Time (yr)")

    ax.set_yticklabels([])


def plot_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    mean_e = []
    # min_e = []
    # max_e = []

    time = []
    for step in steps:
        time.extend(step["time"])
        data = step[column][:, 1:]
        std_e.extend(np.nanstd(data, axis=1))
        mean_e.extend(np.nanmean(data, axis=1))
        # min_e.extend(np.percentile(data, 25, axis=1))
        # max_e.extend(np.percentile(data, 75, axis=1))

    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - np.array(std_e), np.array(mean_e) + np.array(std_e), alpha=0.1, ec="face")
    # ax.fill_between(time, np.array(min_e), np.array(max_e), alpha=0.1, ec="face")

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")
    

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        data = step[column][:, 1:]
        std_e.extend(np.nanstd(data, axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext, **kwargs)
    plot_std_over_time(axs[1], steps, column, mathtext, **kwargs)


fig, axs = plt.subplots(4, 2, figsize=(12, 20), constrained_layout=True)

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        plot_remaining_over_time(axs[0][0], sorted_steps, label=make_label(k))
        plot_energy_over_time(axs[0][1], sorted_steps, label=make_label(k))
        double_plot(axs[1], sorted_steps, "a", "a", label=make_label(k))
        double_plot(axs[2], sorted_steps, "ecc", "e", label=make_label(k))
        double_plot(axs[3], sorted_steps, "inc", "i", label=make_label(k))

fig.suptitle("Time evolution of orbits (N=100)")

plt.figlegend(handles=axs[1][0].get_lines(), ncol=4, loc="upper center", frameon=True)

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_remaining_over_time(ax, steps, **kwargs):
    remaining = []
    time = []
    for step in steps:
        time.extend(step["time"])
        remaining.extend(np.sum(~np.isnan(step["mass"]), axis=1) - 1)

    ax.plot(time, np.array(remaining), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"Remaining objects")
    ax.set_xlabel("Time (yr)")

    ax.set_ylim(0, 105)

def plot_energy_over_time(ax, steps, **kwargs):
    E = []
    time = []
    for step in steps:
        time.extend(step["time"])
        E.extend(step["energy"])

    ax.plot(time, np.array(E), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$E$")
    ax.set_xlabel("Time (yr)")

    ax.set_yticklabels([])


def plot_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    mean_e = []
    # min_e = []
    # max_e = []

    time = []
    for step in steps:
        time.extend(step["time"])
        data = step[column][:, 1:]
        std_e.extend(np.nanstd(data, axis=1))
        mean_e.extend(np.nanmean(data, axis=1))
        # min_e.extend(np.percentile(data, 25, axis=1))
        # max_e.extend(np.percentile(data, 75, axis=1))

    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - np.array(std_e), np.array(mean_e) + np.array(std_e), alpha=0.1, ec="face")
    # ax.fill_between(time, np.array(min_e), np.array(max_e), alpha=0.1, ec="face")

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")
    

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        data = step[column][:, 1:]
        std_e.extend(np.nanstd(data, axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext, **kwargs)
    plot_std_over_time(axs[1], steps, column, mathtext, **kwargs)


fig, axs = plt.subplots(4, 2, figsize=(12, 20), constrained_layout=True)

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        plot_remaining_over_time(axs[0][0], sorted_steps, label=make_label(k))
        plot_energy_over_time(axs[0][1], sorted_steps, label=make_label(k))
        double_plot(axs[1], sorted_steps, "a", "a", label=make_label(k))
        double_plot(axs[2], sorted_steps, "ecc", "e", label=make_label(k))
        double_plot(axs[3], sorted_steps, "inc", "i", label=make_label(k))

fig.suptitle("Time evolution of orbits (N=100)")

plt.figlegend(handles=axs[0][0].get_lines(), ncol=4, loc="upper center", frameon=True)

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_remaining_over_time(ax, steps, **kwargs):
    remaining = []
    time = []
    for step in steps:
        time.extend(step["time"])
        remaining.extend(np.sum(~np.isnan(step["mass"]), axis=1) - 1)

    ax.plot(time, np.array(remaining), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"Remaining objects")
    ax.set_xlabel("Time (yr)")

    ax.set_ylim(0, 105)

def plot_energy_over_time(ax, steps, **kwargs):
    E = []
    time = []
    for step in steps:
        time.extend(step["time"])
        E.extend(step["energy"])

    ax.plot(time, np.array(E), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"Energy")
    ax.set_xlabel("Time (yr)")

    ax.set_yticklabels([])


def plot_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    mean_e = []
    # min_e = []
    # max_e = []

    time = []
    for step in steps:
        time.extend(step["time"])
        data = step[column][:, 1:]
        std_e.extend(np.nanstd(data, axis=1))
        mean_e.extend(np.nanmean(data, axis=1))
        # min_e.extend(np.percentile(data, 25, axis=1))
        # max_e.extend(np.percentile(data, 75, axis=1))

    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - np.array(std_e), np.array(mean_e) + np.array(std_e), alpha=0.1, ec="face")
    # ax.fill_between(time, np.array(min_e), np.array(max_e), alpha=0.1, ec="face")

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")
    

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        data = step[column][:, 1:]
        std_e.extend(np.nanstd(data, axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext, **kwargs)
    plot_std_over_time(axs[1], steps, column, mathtext, **kwargs)


fig, axs = plt.subplots(4, 2, figsize=(12, 20), constrained_layout=True)

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        plot_remaining_over_time(axs[0][0], sorted_steps, label=make_label(k))
        plot_energy_over_time(axs[0][1], sorted_steps, label=make_label(k))
        double_plot(axs[1], sorted_steps, "a", "a", label=make_label(k))
        double_plot(axs[2], sorted_steps, "ecc", "e", label=make_label(k))
        double_plot(axs[3], sorted_steps, "inc", "i", label=make_label(k))

fig.suptitle("Time evolution of orbits (N=100)")

plt.figlegend(handles=axs[0][0].get_lines(), ncol=4, loc="upper center", frameon=True)

# %%
import h5py
import util

def make_label(k):
    if k[0] == 1:
        return f"Seed = {k[2]}, Drag"
    if k[0] == 0:
        return f"Seed = {k[2]}, No Drag"

def plot_remaining_over_time(ax, steps, **kwargs):
    remaining = []
    time = []
    for step in steps:
        time.extend(step["time"])
        remaining.extend(np.sum(~np.isnan(step["mass"]), axis=1) - 1)

    ax.plot(time, np.array(remaining), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"Remaining objects")
    ax.set_xlabel("Time (yr)")

    ax.set_ylim(0, 105)

def plot_energy_over_time(ax, steps, **kwargs):
    E = []
    time = []
    for step in steps:
        time.extend(step["time"])
        E.extend(step["energy"])

    ax.plot(time, np.array(E), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"Energy")
    ax.set_xlabel("Time (yr)")

    ax.set_yticklabels([])


def plot_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    mean_e = []
    # min_e = []
    # max_e = []

    time = []
    for step in steps:
        time.extend(step["time"])
        data = step[column][:, 1:]
        std_e.extend(np.nanstd(data, axis=1))
        mean_e.extend(np.nanmean(data, axis=1))
        # min_e.extend(np.percentile(data, 25, axis=1))
        # max_e.extend(np.percentile(data, 75, axis=1))

    ax.plot(time, np.array(mean_e), **kwargs)
    ax.fill_between(time, np.array(mean_e) - np.array(std_e), np.array(mean_e) + np.array(std_e), alpha=0.1, ec="face")
    # ax.fill_between(time, np.array(min_e), np.array(max_e), alpha=0.1, ec="face")

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{mean}}({mathtext}) \pm \mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")
    

def plot_std_over_time(ax, steps, column, mathtext, **kwargs):
    std_e = []
    time = []
    for step in steps:
        time.extend(step["time"])
        data = step[column][:, 1:]
        std_e.extend(np.nanstd(data, axis=1))
    ax.plot(time, np.array(std_e), **kwargs)

    # ax.legend()
    ax.set_xlim(0, 100000)
    ax.set_title(rf"$\mathrm{{std}}({mathtext})$")
    ax.set_xlabel("Time (yr)")


def double_plot(axs, steps, column, mathtext, **kwargs):
    plot_over_time(axs[0], steps, column, mathtext, **kwargs)
    plot_std_over_time(axs[1], steps, column, mathtext, **kwargs)


fig, axs = plt.subplots(4, 2, figsize=(12, 20), constrained_layout=True)

for k, v in sims.items():
    if v["n_particles"] > 50:
        f = h5py.File(v["path"] / f"data_reb_{k[2]}.h5", "r")
        sorted_steps = util.sorted_steps(f)
        plot_remaining_over_time(axs[0][0], sorted_steps, label=make_label(k))
        plot_energy_over_time(axs[0][1], sorted_steps, label=make_label(k))
        double_plot(axs[1], sorted_steps, "a", "a", label=make_label(k))
        double_plot(axs[2], sorted_steps, "ecc", "e", label=make_label(k))
        double_plot(axs[3], sorted_steps, "inc", "i", label=make_label(k))

fig.suptitle("Time evolution of orbits (N=100)", alpha=0)

plt.figlegend(handles=axs[0][0].get_lines(), ncol=4, loc="upper center", frameon=True)

# %%
import matplotlib.pyplot as plt

for k, v in sims.items():
    if v["n_particles"] > 50 and v["drag_coefficient"] == 0:
        runtimes = v["runtimes"].T
        runtimes2 = sims[(1, k[1], k[2])]["runtimes"].T

        Ns = []
        for N in np.arange(runtimes[0, 0], runtimes[0, -1] - 1, -1):
            Ns.append((N - 1, np.median(runtimes[1, runtimes[0] == N]) / np.median(runtimes2[1, runtimes2[0] == N])))
        Ns = np.array(Ns).T
        plt.plot(Ns[0], Ns[1])

plt.ylabel("Runtime speed ratio")
plt.xlabel("Planets remaining")

plt.xlim(100, 15)

plt.show()

# %%
import matplotlib.pyplot as plt

for k, v in sims.items():
    if v["n_particles"] < 50:
        runtimes = v["runtimes"].T

        Ns = []
        for N in np.arange(runtimes[0, 0], runtimes[0, -1] - 1, -1):
            Ns.append((N - 1, np.mean(runtimes[1, runtimes[0] == N])))
        Ns = np.array(Ns).T
        plt.plot(Ns[0], Ns[1], label=k)

plt.yscale("log")
plt.ylabel("Simulation Years / Second of runtime")
plt.xlabel("Planets remaining")
plt.xlim(100, 15)
plt.legend()
plt.show()

# %%
import matplotlib.pyplot as plt

for k, v in sims.items():
    if v["n_particles"] < 50:
        runtimes = v["runtimes"].T

        Ns = []
        for N in np.arange(runtimes[0, 0], runtimes[0, -1] - 1, -1):
            Ns.append((N - 1, np.mean(runtimes[1, runtimes[0] == N])))
        Ns = np.array(Ns).T
        plt.plot(Ns[0], Ns[1], label=k)

plt.yscale("log")
plt.ylabel("Simulation Years / Second of runtime")
plt.xlabel("Planets remaining")
plt.xlim(25, 15)
plt.legend()
plt.show()

# %%
import matplotlib.pyplot as plt

for k, v in sims.items():
    if v["n_particles"] < 50:
        runtimes = v["runtimes"].T

        Ns = []
        for N in np.arange(runtimes[0, 0], runtimes[0, -1] - 1, -1):
            Ns.append((N - 1, np.mean(runtimes[1, runtimes[0] == N])))
        Ns = np.array(Ns).T
        plt.plot(Ns[0], Ns[1], label=k)

# plt.yscale("log")
plt.ylabel("Simulation Years / Second of runtime")
plt.xlabel("Planets remaining")
plt.xlim(25, 15)
plt.legend()
plt.show()

# %%
import matplotlib.pyplot as plt

for k, v in sims.items():
    if v["n_particles"] < 50:
        runtimes = v["runtimes"].T

        Ns = []
        for N in np.arange(runtimes[0, 0], runtimes[0, -1] - 1, -1):
            Ns.append((N - 1, np.mean(runtimes[1, runtimes[0] == N])))
        Ns = np.array(Ns).T
        plt.plot(Ns[0], Ns[1], label=k)

plt.yscale("log")
plt.ylabel("Simulation Years / Second of runtime")
plt.xlabel("Planets remaining")
plt.xlim(25, 0)
plt.legend()
plt.show()

# %%
import matplotlib.pyplot as plt

for k, v in sims.items():
    if v["n_particles"] < 50:
        runtimes = v["runtimes"].T

        Ns = []
        for N in np.arange(runtimes[0, 0], runtimes[0, -1] - 1, -1):
            Ns.append((N - 1, np.mean(runtimes[1, runtimes[0] == N])))
        Ns = np.array(Ns).T
        plt.plot(Ns[0], Ns[1], label=k)

# plt.yscale("log")
plt.ylabel("Simulation Years / Second of runtime")
plt.xlabel("Planets remaining")
plt.xlim(25, 0)
plt.legend()
plt.show()

# %%
import matplotlib.pyplot as plt

for k, v in sims.items():
    if v["n_particles"] < 50:
        runtimes = v["runtimes"].T

        Ns = []
        for N in np.arange(runtimes[0, 0], runtimes[0, -1] - 1, -1):
            Ns.append((N - 1, np.mean(runtimes[1, runtimes[0] == N])))
        Ns = np.array(Ns).T
        plt.plot(Ns[0], Ns[1]*60, label=k)

# plt.yscale("log")
plt.ylabel("Simulation Years / Minute of runtime")
plt.xlabel("Planets remaining")
plt.xlim(25, 0)
plt.legend()
plt.show()

# %%
import matplotlib.pyplot as plt

for k, v in sims.items():
    if v["n_particles"] > 50:
        runtimes = v["runtimes"].T

        Ns = []
        for N in np.arange(runtimes[0, 0], runtimes[0, -1] - 1, -1):
            Ns.append((N - 1, np.mean(runtimes[1, runtimes[0] == N])))
        Ns = np.array(Ns).T
        plt.plot(Ns[0], Ns[1]*60, label=k)

plt.yscale("log")
plt.ylabel("Simulation Years / Minute of runtime")
plt.xlabel("Planets remaining")
plt.xlim(100, 15)
plt.legend()
plt.show()

# %%
import matplotlib.pyplot as plt

for k, v in sims.items():
    if v["n_particles"] < 50 and v["drag_coefficient"] == 0:
        runtimes = v["runtimes"].T
        runtimes2 = sims[(1, k[1], k[2])]["runtimes"].T

        Ns = []
        for N in np.arange(runtimes[0, 0], runtimes[0, -1] - 1, -1):
            Ns.append((N - 1, np.median(runtimes[1, runtimes[0] == N]) / np.median(runtimes2[1, runtimes2[0] == N])))
        Ns = np.array(Ns).T
        plt.plot(Ns[0], Ns[1])

plt.ylabel("Runtime speed ratio")
plt.xlabel("Planets remaining")

plt.xlim(25, 0)

plt.show()

# %%
import matplotlib.pyplot as plt

for k, v in sims.items():
    if v["n_particles"] > 0:
        runtimes = v["runtimes"].T

        Ns = []
        for N in np.arange(runtimes[0, 0], runtimes[0, -1] - 1, -1):
            Ns.append((N - 1, np.mean(runtimes[1, runtimes[0] == N])))
        Ns = np.array(Ns).T
        plt.plot(Ns[0], Ns[1]*60, label=k)

plt.yscale("log")
plt.ylabel("Simulation Years / Minute of runtime")
plt.xlabel("Planets remaining")
plt.xlim(100, 15)
plt.legend()
plt.show()

# %%
import matplotlib.pyplot as plt

for k, v in sims.items():
    if v["n_particles"] > 0:
        runtimes = v["runtimes"].T

        Ns = []
        for N in np.arange(runtimes[0, 0], runtimes[0, -1] - 1, -1):
            Ns.append((N - 1, np.mean(runtimes[1, runtimes[0] == N])))
        Ns = np.array(Ns).T
        plt.plot(Ns[0], Ns[1]*60, label=k)

plt.yscale("log")
plt.ylabel("Simulation Years / Minute of runtime")
plt.xlabel("Planets remaining")
plt.xlim(40, 0)
plt.legend()
plt.show()

# %%
import matplotlib.pyplot as plt

for k, v in sims.items():
    if v["n_particles"] > 0:
        runtimes = v["runtimes"].T

        Ns = []
        for N in np.arange(runtimes[0, 0], runtimes[0, -1] - 1, -1):
            Ns.append((N - 1, np.mean(runtimes[1, runtimes[0] == N])))
        Ns = np.array(Ns).T
        plt.plot(Ns[0], Ns[1]*60, label=k)

# plt.yscale("log")
plt.ylabel("Simulation Years / Minute of runtime")
plt.xlabel("Planets remaining")
plt.xlim(40, 0)
plt.legend()
plt.show()

# %%
import matplotlib.pyplot as plt

for k, v in sims.items():
    if v["n_particles"] > 0 and v["drag_coefficient"] == 1:
        runtimes = v["runtimes"].T

        Ns = []
        for N in np.arange(runtimes[0, 0], runtimes[0, -1] - 1, -1):
            Ns.append((N - 1, np.mean(runtimes[1, runtimes[0] == N])))
        Ns = np.array(Ns).T
        plt.plot(Ns[0], Ns[1]*60, label=k)

# plt.yscale("log")
plt.ylabel("Simulation Years / Minute of runtime")
plt.xlabel("Planets remaining")
plt.xlim(40, 0)
plt.legend()
plt.show()

# %%
import matplotlib.pyplot as plt

for k, v in sims.items():
    if v["n_particles"] > 50 and v["drag_coefficient"] == 0:
        runtimes = v["runtimes"].T
        runtimes2 = sims[(1, k[1], k[2])]["runtimes"].T

        Ns = []
        for N in np.arange(runtimes[0, 0], runtimes[0, -1] - 1, -1):
            Ns.append((N - 1, np.median(runtimes[1, runtimes[0] == N]) / np.median(runtimes2[1, runtimes2[0] == N])))
        Ns = np.array(Ns).T
        plt.plot(Ns[0], Ns[1])

plt.ylabel("Runtime speed ratio")
plt.xlabel("Planets remaining")

plt.xlim(100, 15)
plt.xlim()

plt.show()

# %%
import matplotlib.pyplot as plt

for k, v in sims.items():
    if v["n_particles"] > 50 and v["drag_coefficient"] == 0:
        runtimes = v["runtimes"].T
        runtimes2 = sims[(1, k[1], k[2])]["runtimes"].T

        Ns = []
        for N in np.arange(runtimes[0, 0], runtimes[0, -1] - 1, -1):
            Ns.append((N - 1, np.median(runtimes[1, runtimes[0] == N]) / np.median(runtimes2[1, runtimes2[0] == N])))
        Ns = np.array(Ns).T
        plt.plot(Ns[0], Ns[1])

plt.ylabel("Runtime speed ratio")
plt.xlabel("Planets remaining")

plt.xlim(100, 15)
plt.y(1.0, 2.0)

plt.show()

# %%
import matplotlib.pyplot as plt

for k, v in sims.items():
    if v["n_particles"] > 50 and v["drag_coefficient"] == 0:
        runtimes = v["runtimes"].T
        runtimes2 = sims[(1, k[1], k[2])]["runtimes"].T

        Ns = []
        for N in np.arange(runtimes[0, 0], runtimes[0, -1] - 1, -1):
            Ns.append((N - 1, np.median(runtimes[1, runtimes[0] == N]) / np.median(runtimes2[1, runtimes2[0] == N])))
        Ns = np.array(Ns).T
        plt.plot(Ns[0], Ns[1])

plt.ylabel("Runtime speed ratio")
plt.xlabel("Planets remaining")

plt.xlim(100, 15)
plt.ylim(1.0, 2.0)

plt.show()

# %%
import matplotlib.pyplot as plt

for k, v in sims.items():
    if v["n_particles"] < 50 and v["drag_coefficient"] == 0:
        runtimes = v["runtimes"].T
        runtimes2 = sims[(1, k[1], k[2])]["runtimes"].T

        Ns = []
        for N in np.arange(runtimes[0, 0], runtimes[0, -1] - 1, -1):
            Ns.append((N - 1, np.median(runtimes[1, runtimes[0] == N]) / np.median(runtimes2[1, runtimes2[0] == N])))
        Ns = np.array(Ns).T
        plt.plot(Ns[0], Ns[1])

plt.ylabel("Runtime speed ratio")
plt.xlabel("Planets remaining")

plt.xlim(100, 15)
plt.ylim(1.0, 2.0)

plt.show()

# %%
import matplotlib.pyplot as plt

for k, v in sims.items():
    if v["n_particles"] > 50 and v["drag_coefficient"] == 0:
        runtimes = v["runtimes"].T
        runtimes2 = sims[(1, k[1], k[2])]["runtimes"].T

        Ns = []
        for N in np.arange(runtimes[0, 0], runtimes[0, -1] - 1, -1):
            Ns.append((N - 1, np.median(runtimes[1, runtimes[0] == N]) / np.median(runtimes2[1, runtimes2[0] == N])))
        Ns = np.array(Ns).T
        plt.plot(Ns[0], Ns[1])

plt.ylabel("Runtime speed ratio")
plt.xlabel("Planets remaining")

plt.xlim(100, 15)
plt.ylim(1.0, 2.0)

plt.show()

# %%
import matplotlib.pyplot as plt

for k, v in sims.items():
    if v["n_particles"] > 0 and v["drag_coefficient"] == 1:
        runtimes = v["runtimes"].T

        Ns = []
        for N in np.arange(runtimes[0, 0], runtimes[0, -1] - 1, -1):
            Ns.append((N - 1, np.mean(runtimes[1, runtimes[0] == N])))
        Ns = np.array(Ns).T
        plt.plot(Ns[0], Ns[1]*60, label=k)

# plt.yscale("log")
plt.ylabel("Simulation Years / Minute of runtime")
plt.xlabel("Planets remaining")
plt.xlim(100, 0)
plt.legend()
plt.show()

# %%
import matplotlib.pyplot as plt

for k, v in sims.items():
    if v["n_particles"] > 0 and v["drag_coefficient"] == 1:
        runtimes = v["runtimes"].T

        Ns = []
        for N in np.arange(runtimes[0, 0], runtimes[0, -1] - 1, -1):
            Ns.append((N - 1, np.mean(runtimes[1, runtimes[0] == N])))
        Ns = np.array(Ns).T
        plt.plot(Ns[0], Ns[1]*60, label=k)

plt.yscale("log")
plt.ylabel("Simulation Years / Minute of runtime")
plt.xlabel("Planets remaining")
plt.xlim(100, 0)
plt.legend()
plt.show()

# %%
import matplotlib.pyplot as plt

for k, v in sims.items():
    if v["n_particles"] > 0 and v["drag_coefficient"] == 1:
        runtimes = v["runtimes"].T

        Ns = []
        for N in np.arange(runtimes[0, 0], runtimes[0, -1] - 1, -1):
            Ns.append((N - 1, np.mean(runtimes[1, runtimes[0] == N])))
        Ns = np.array(Ns).T
        plt.plot(Ns[0], Ns[1]*60, label=k)

plt.yscale("log")
plt.ylabel("Simulation Years / Minute of runtime")
plt.xlabel("Planets remaining")
plt.xlim(100, 0)
plt.ylim(1, 2000)
plt.legend()
plt.show()

# %%
import matplotlib.pyplot as plt

for k, v in sims.items():
    if v["n_particles"] > 0 and v["drag_coefficient"] == 1:
        runtimes = v["runtimes"].T

        Ns = []
        for N in np.arange(runtimes[0, 0], runtimes[0, -1] - 1, -1):
            Ns.append((N - 1, np.mean(runtimes[1, runtimes[0] == N])))
        Ns = np.array(Ns).T
        plt.plot(Ns[0], Ns[1]*60, label=k)

plt.yscale("log")
plt.ylabel("Simulation Years / Minute of runtime")
plt.xlabel("Planets remaining")
plt.xlim(100, 0)
plt.ylim(5, 2000)
plt.legend()
plt.show()

# %%
import matplotlib.pyplot as plt

for k, v in sims.items():
    if v["n_particles"] > 0:
        runtimes = v["runtimes"].T

        Ns = []
        for N in np.arange(runtimes[0, 0], runtimes[0, -1] - 1, -1):
            Ns.append((N - 1, np.mean(runtimes[1, runtimes[0] == N])))
        Ns = np.array(Ns).T
        plt.plot(Ns[0], Ns[1]*60, label=k)

plt.yscale("log")
plt.ylabel("Simulation Years / Minute of runtime")
plt.xlabel("Planets remaining")
plt.xlim(100, 0)
plt.ylim(5, 2000)
plt.legend()
plt.show()

# %%
import matplotlib.pyplot as plt

for k, v in sims.items():
    if v["n_particles"] > 0 and v["drag_coefficient"] == 1:
        runtimes = v["runtimes"].T

        Ns = []
        for N in np.arange(runtimes[0, 0], runtimes[0, -1] - 1, -1):
            Ns.append((N - 1, np.mean(runtimes[1, runtimes[0] == N])))
        Ns = np.array(Ns).T
        plt.plot(Ns[0], Ns[1]*60, label=k)

plt.yscale("log")
plt.ylabel("Simulation Years / Minute of runtime")
plt.xlabel("Planets remaining")
plt.xlim(100, 0)
plt.ylim(5, 2000)
plt.legend()
plt.show()

# %%
import matplotlib.pyplot as plt

for k, v in sims.items():
    if v["n_particles"] > 50 and v["drag_coefficient"] == 1:
        runtimes = v["runtimes"].T

        Ns = []
        for N in np.arange(runtimes[0, 0], runtimes[0, -1] - 1, -1):
            Ns.append((N - 1, np.mean(runtimes[1, runtimes[0] == N])))
        Ns = np.array(Ns).T
        plt.plot(Ns[0], Ns[1]*60, label=k)

plt.yscale("log")
plt.ylabel("Simulation Years / Minute of runtime")
plt.xlabel("Planets remaining")
plt.xlim(100, 0)
plt.ylim(5, 2000)
plt.legend()
plt.show()

# %%
import matplotlib.pyplot as plt

for k, v in sims.items():
    if v["n_particles"] > 50J
        runtimes = v["runtimes"].T

        Ns = []
        for N in np.arange(runtimes[0, 0], runtimes[0, -1] - 1, -1):
            Ns.append((N - 1, np.mean(runtimes[1, runtimes[0] == N])))
        Ns = np.array(Ns).T
        plt.plot(Ns[0], Ns[1]*60, label=k)

plt.yscale("log")
plt.ylabel("Simulation Years / Minute of runtime")
plt.xlabel("Planets remaining")
plt.xlim(100, 0)
plt.ylim(5, 2000)
plt.legend()
plt.show()

# %%
import matplotlib.pyplot as plt

for k, v in sims.items():
    if v["n_particles"] > 50:
        runtimes = v["runtimes"].T

        Ns = []
        for N in np.arange(runtimes[0, 0], runtimes[0, -1] - 1, -1):
            Ns.append((N - 1, np.mean(runtimes[1, runtimes[0] == N])))
        Ns = np.array(Ns).T
        plt.plot(Ns[0], Ns[1]*60, label=k)

plt.yscale("log")
plt.ylabel("Simulation Years / Minute of runtime")
plt.xlabel("Planets remaining")
plt.xlim(100, 0)
plt.ylim(5, 2000)
plt.legend()
plt.show()

# %%
import matplotlib.pyplot as plt

for k, v in sims.items():
    if v["n_particles"] > 50:
        runtimes = v["runtimes"].T

        Ns = []
        for N in np.arange(runtimes[0, 0], runtimes[0, -1] - 1, -1):
            Ns.append((N - 1, np.mean(runtimes[1, runtimes[0] == N])))
        Ns = np.array(Ns).T
        plt.plot(Ns[0], Ns[1]*60, label=make_label(k))

plt.yscale("log")
plt.ylabel("Simulation Years / Minute of runtime")
plt.xlabel("Planets remaining")
plt.xlim(100, 0)
plt.ylim(5, 2000)
plt.legend()
plt.show()

# %%
import matplotlib.pyplot as plt

for k, v in sims.items():
    if v["n_particles"] > 50 and v["drag_coefficient"] == 0:
        runtimes = v["runtimes"].T
        runtimes2 = sims[(1, k[1], k[2])]["runtimes"].T

        Ns = []
        for N in np.arange(runtimes[0, 0], runtimes[0, -1] - 1, -1):
            Ns.append((N - 1, np.median(runtimes[1, runtimes[0] == N]) / np.median(runtimes2[1, runtimes2[0] == N])))
        Ns = np.array(Ns).T
        plt.plot(Ns[0], Ns[1])

plt.ylabel("Runtime speed ratio")
plt.xlabel("Planets remaining")

plt.xlim(100, 15)

plt.show()

# %%
