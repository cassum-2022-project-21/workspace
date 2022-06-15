import os
import numpy as np
from SiMon.ic_generator import InitialConditionGenerator
from itertools import product
from random import sample

def gen_seeds():
    return [38394, 47220, 61392, 62232, 69762, 812, 83482, 85197, 87199, 92809, 25452, 4901, 75344, 52174, 11564, 2929, 45297, 61949, 58910, 14175]

def generate_ic():
    # parameter_space = {
    #     "random-seed": gen_seeds,
    #     "t_end": 200000.0,
    #     "N_end": 2,
    #     "N_enddelay": 5000.0,
    #     "store-dt": 1.0,
    #     "n-particles": 25,
    #     "code-name": "rebound",
    #     "alpha": [None, -2],
    #     # "rebound-archive": "rebound_archive.bin",
    #     ("pa-rate", "pa-beta"): [(0.0, None), (1e-11, "2_3")]
    # }

    # parameter_space = {
    #     "random-seed": 14175,
    #     "t_end": 200000.0,
    #     "N_end": 2,
    #     "N_enddelay": 5000.0,
    #     "store-dt": 1.0,
    #     "n-particles": 25,
    #     "code-name": "rebound",
    #     "alpha": [-2],
    #     # "rebound-archive": "rebound_archive.bin",
    #     ("pa-rate", "pa-beta"): [(0.0, None)]
    # }

    parameter_space = {
        "random-seed": [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000],
        "t_end": 200000.0,
        "N_end": 2,
        "N_enddelay": 2000.0,
        "store-dt": 1.0,
        "n-particles": 2,
        "alpha": -2,
        "code-name": "rebound",
        "rebound-archive": "rebound_archive.bin",
        ("pa-rate", "pa-beta"): (0.0, None),

        "m-total": 1.0,
        "a-in": 0.23,
        "a-out": 0.24,
        "std-e": 0.1,

        "drag-coefficient": [0.0, 1.0],
        "velocity-file": "/home/alchzh/cassum-2022-project-21/workspace/disk/velocity.txt",
        "density-file": "/home/alchzh/cassum-2022-project-21/workspace/disk/midplane_density_2.txt"
    }

    # templates
    code_name = "IOPFSimulation"
    executable_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "simulation.py"
    )
    start_cmd_template = "rm -f DONE ; python -u {executable_path} {args} 1>>output.txt 2>>error.txt"
    restart_cmd_template = "rm -f DONE ; touch restart.txt ; python -u {executable_path} {args} 1>>output.txt 2>>error.txt"
    stop_cmd = "touch STOP"
    output_dir_template = "iopf_sim_DRAG_{drag_coefficient}__ALPHA_{alpha}_BETA_{pa_beta}_{random_seed}"

    # IC generator
    ic = InitialConditionGenerator(conf_file="SiMon.conf")

    param_names = []
    value_lists = []
    for p, v in parameter_space.items():
        if callable(v):
            v = v()
        if not isinstance(v, list):
            v = [v]
        param_names.append(p)
        value_lists.append(v)
    for values in product(*value_lists):
        args_list = []
        args_dict = {}
        for i, value in enumerate(values):
            param_name = param_names[i]
            if isinstance(param_name, str):
                if value is not None:
                    args_list.append(f"--{param_name}")
                    args_list.append(str(value))
                args_dict[param_name.replace("-", "_")] = value
            elif isinstance(param_name, tuple):
                for j, p in enumerate(param_name):
                    if value[j] is not None:
                        args_list.append(f"--{p}")
                        args_list.append(str(value[j]))
                    args_dict[p.replace("-", "_")] = value[j]
            else:
                raise TypeError("Invalid parameter type")
        args = " ".join(args_list)
        start_cmd = start_cmd_template.format(
            executable_path=executable_path,
            args=args,
            **args_dict
        )
        restart_cmd = restart_cmd_template.format(
            executable_path=executable_path,
            args=args,
            **args_dict
        )
        output_dir = output_dir_template.format(
            executable_path=executable_path,
            args=args,
            **args_dict
        )

        ic.generate_simulation_ic(
            code_name,
            parameter_space["t_end"],
            output_dir,
            start_cmd,
            input_file="input.txt",
            output_file="output.txt",
            error_file="error.txt",
            restart_cmd=restart_cmd,
            stop_cmd=stop_cmd,
        )


if __name__ == "__main__":
    generate_ic()
