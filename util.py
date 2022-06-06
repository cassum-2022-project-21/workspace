from collections import namedtuple
import matplotlib.pyplot as plt
from pathlib import Path

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
