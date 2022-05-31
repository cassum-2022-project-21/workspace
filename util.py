import matplotlib.pyplot as plt

def prop_table(objs, prop_cycle=None):
    """Returns a table for an ordered collection based on prop cycle"""
    if prop_cycle is None:
        prop_cycle = plt.rcParams["axes.prop_cycle"]
    c = prop_cycle()
    return {o: next(c) for o in objs}

def sorted_steps(dict_like):
    """Returns a sorted list of the "Step#" groups in a h5 file (or other dict-like object)"""
    return [dict_like[f"Step#{i}"] for i in sorted(int(group[5:]) for group in dict_like)]
