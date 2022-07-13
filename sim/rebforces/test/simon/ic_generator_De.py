import os
import numpy as np
from SiMon.ic_generator import InitialConditionGenerator
from itertools import product
from random import sample

def generate_ic_single():
    parameter_space = {
        "m": [f"{k:.1f}" for k in np.linspace(-3, 0, 11)],
        "a": 0.30,
        "e": list(np.linspace(0, 0.1, 11)),
        "C_D": 1.0,
        "eta": 0.002,
        "t_end": 10000.0
    }

    # templates
    code_name = "IOPFCSimulation"
    executable_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../test-scaled-keplerian-drag"
    )
    start_cmd_template = "rm -f DONE ; {executable_path} {args} 1>>output.txt 2>>error.txt"
    restart_cmd_template = "touch restart.txt ; {executable_path} {args} 1>>output.txt 2>>error.txt"
    stop_cmd = "touch STOP"
    output_dir_template = "iopf_csim_DRAG_{m}_{a}_{e}_{C_D}"

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
                    # args_list.append(f"--{param_name}")
                    args_list.append(str(value))
                args_dict[param_name.replace("-", "_")] = value
            elif isinstance(param_name, tuple):
                for j, p in enumerate(param_name):
                    if value[j] is not None:
                        # args_list.append(f"--{p}")
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
            100000,
            output_dir,
            start_cmd,
            input_file="input.txt",
            output_file="output.txt",
            error_file="error.txt",
            restart_cmd=restart_cmd,
            stop_cmd=stop_cmd,
        )


if __name__ == "__main__":
    generate_ic_single()
