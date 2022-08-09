import os
import numpy as np
from itertools import product
from random import sample

# k = [6674, 11857, 17619, 49863, 70080, 72049, 80151, 95704, 96616]

k = [70080, 72049, 80151, 95704, 96616]

def generate_ic():
    parameter_space = {
        "random-seed": k,
        "t_end": 2000000.0,
        "store-dt": 1000.0,
        "n-particles": 100,
        "code-name": "rebound",
        "alpha": -2,
        # "rebound-archive": "rebound_archive.bin",
        ("pa-rate", "pa-beta"): [(0.0, None)],
        "drag-coefficient": 1.0,
        "migration-torque": "",

        "N_end": 1,
        "N_enddelay": 1.0,

        "m-total": 1.0,

        # ("a-in", "a-out", "std-e", "std-i"): [(0.22, 0.24, 0.02, 0.01)],
        # ("a-in", "a-out", "std-e", "std-i"): [(0.229, 0.231, 0.02, 0.0)],
        ("a-in", "a-out", "std-e", "std-i"): [(0.229, 0.231, 0.02, 0.01), (0.22, 0.24, 0.02, 0.0)],

        "rho": 3.0,
        "N_handoff": 45

        # "continue-from": 1100000.0
    }

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

    # parameter_space = {
    #     "random-seed": gen_seeds(),
    #     "t_end": 3500000.0,
    #     "store-dt": 100.0,
    #     "n-particles": 100,
    #     "alpha": -2,
    #     "code-name": "rebound",
    #     "rebound-archive": "rebound_archive.bin",
    #     ("pa-rate", "pa-beta"): (0.0, None),

    #     "m-total": 1.0,
    #     "a-in": 0.22,
    #     "a-out": 0.24,
    #     "rho": 3.0,

    #     "drag-coefficient": [0.0, 1.0]
    # }

    # templates
    code_name = "IOPFSimulation"
    executable_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "simulation_combined.py"
    )
    start_cmd_template = "rm -f DONE ; python -u {executable_path} {args} 1>>output.txt 2>>error.txt"
    restart_cmd_template = "rm -f DONE ; touch restart.txt ; python -u {executable_path} {args} 1>>output.txt 2>>error.txt"
    stop_cmd = "touch STOP"
    output_dir_template = "iopf_sim_i_{std_i}_a_{a_in}_{a_out}_{random_seed}"
    # output_dir_template = "iopf_old_sim_NODRAG_BETA_{pa_beta}_N_{n_particles}_{random_seed}"

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

        if "NEW_SLURM_JOB_NAME" in os.environ:
            from pathlib import Path

            jobdir = Path(os.environ["PARENT_SLURM_DIR"]) / os.environ["NEW_SLURM_JOB_NAME"] / output_dir
            jobdir.mkdir(parents=True, exist_ok=True)

            with open(jobdir / "exec.sh", "w") as f:
                f.write(start_cmd)
            
        else:
            from astrosimon import InitialConditionGenerator

            # IC generator
            ic = InitialConditionGenerator(conf_file="SiMon.conf")

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

    if "NEW_SLURM_JOB_NAME" in os.environ:
        p = Path(os.environ["PARENT_SLURM_DIR"]) / os.environ["NEW_SLURM_JOB_NAME"]
        
        with open(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "vera_script.sh"
        ), "r") as src, open(p / "vera_script.sh", "w") as dest:
            s = src.read().replace("<<<JOB_NAME>>>", os.environ["NEW_SLURM_JOB_NAME"])
            dest.write(s)
            

if __name__ == "__main__":
    generate_ic()
