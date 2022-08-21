from pathlib import Path
from glob import glob
import subprocess
import re

N_PATTERN = re.compile(r"N = ([0-9]+\.?[0-9]*)")
T_PATTERN = re.compile(r"t = ([0-9]+\.?[0-9]*)")

for output_file in glob("*/output.txt"):
    p = Path(output_file)
    d = p.parent.name
    last_lines = subprocess.check_output(["tail", "-25", output_file]).decode('utf-8')
    N = N_PATTERN.findall(last_lines)[-1][1]
    t = T_PATTERN.findall(last_lines)[-1][1]
    print(f"d: {N=} {t=}")