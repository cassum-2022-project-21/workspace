import os
import subprocess
from pathlib import Path
from ctypes import CDLL


def compile(code, inject, flags=[]):
    pid = os.getpid()
    


