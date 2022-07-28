from collections import namedtuple
import matplotlib.pyplot as plt
from pathlib import Path
import re

def prop_table(objs, prop_cycle=None):
    """Returns a table for an ordered collection based on prop cycle"""
    if prop_cycle is None:
        prop_cycle = plt.rcParams["axes.prop_cycle"]
    c = prop_cycle()
    return {o: next(c) for o in objs}

def sorted_steps(dict_like):
    """Returns a sorted list of the "Step#" groups in a h5 file (or other dict-like object)"""
    return [dict_like[f"Step#{i}"] for i in sorted(int(group[5:]) for group in dict_like)]

ParamSet = namedtuple("ParamSet", ["alpha", "beta", "seed"])
def read_params(name):
    tokens = name.split("_")
    seed = int(tokens[-1])
    alpha = None
    beta = None
    for i, token in enumerate(tokens):
        if token == "ALPHA" and tokens[i+1] != "None":
            alpha = int(tokens[i+1])
        elif token == "BETA" and tokens[i+1] != "None":
            beta = f"{tokens[i+1]}_{tokens[i+2]}"
    return ParamSet(alpha, beta, seed)

def find_all_simulations(dir):
    p = Path(dir)
    return { read_params(h5path.parts[len(p.parts)]): h5path for h5path in p.glob("iopf_sim_*/**/*.h5") }

NUMBER_REGEX = re.compile(r"\-?(?:[0-9]+\.)?[0-9]+(?:e[\+-][0-9]{2})?")

def new_load_simulation(paths, dest=None, prefix=""):
    if dest is None:
        dest = {}

    for sim_path in paths:
        label = sim_path.name
        slabel = label.split("_")
        drag_coefficient = int(slabel[slabel.index('DRAG')+1][0])
        n_particles = int(slabel[slabel.index('N')+1])
        seed = int(slabel[-1])

        label = prefix + label
        dest[label] = {
            "label": label,
            "drag_coefficient": drag_coefficient,
            "n_particles": n_particles,
            "seed": seed,
            "path": sim_path,
            # "runtimes": np.array(L)
        }

    return dest

def new_load_simulation_20220725(paths, dest=None, prefix=""):
    if dest is None:
        dest = {}

    for sim_path in paths:
        label = sim_path.name
        slabel = label.split("_")
        drag_coefficient = 1
        n_particles = 100
        seed = int(slabel[-1])

        if "0.24" in slabel or ("0.01" in slabel and "a" not in slabel):
            width = 0.02
        else:
            width = 0.002

        inc = float(slabel[3])

        label = prefix + label
        dest[label] = {
            "label": label,
            "drag_coefficient": drag_coefficient,
            "n_particles": n_particles,
            "seed": seed,
            "path": sim_path,
            "inc": inc,
            "width": width
            # "runtimes": np.array(L)
        }

    return dest
