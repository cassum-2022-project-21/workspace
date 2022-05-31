from SiMon.simulation import Simulation
import re
import os
import subprocess
import signal

__simulation__ = "IOPFSimulation"

class IOPFSimulation(Simulation):
    def __init__(
        self,
        sim_id,
        name,
        full_dir,
        status,
        mode="daemon",
        t_min=0,
        t_max=0,
        restarts=None,
        logger=None,
    ):
        super().__init__(
            sim_id, name, full_dir, status, mode, t_min, t_max, restarts, logger
        )

    def sim_get_model_time(self):
        super().sim_get_model_time()
        orig_dir = os.getcwd()
        os.chdir(self.full_dir)
        if "Output_file" in self.config:
            output_file = self.config["Output_file"]
            regex = re.compile("\\d+")
            if os.path.isfile(output_file):
                last_line = subprocess.check_output(["tail", "-1", output_file]).decode('utf-8')
                res = regex.findall(last_line)
                if len(res) > 0:
                    self.t = float(res[0])
        os.chdir(orig_dir)
        return self.t

    def sim_stop(self):
        orig_dir = os.getcwd()
        os.chdir(self.full_dir)
        # Find the process by PID
        if os.path.isfile(".process.pid"):
            # if the PID file exists, try to read the process ID
            f_pid = open(".process.pid", "r")
            pid = int(f_pid.readline().strip())
            f_pid.close()
            try:
                os.kill(pid, signal.SIGINT)
                msg = "Simulation %s (PID: %d) SIGINTd." % (self.name, pid)
                print(msg)
                if self.logger is not None:
                    self.logger.info(msg)
            except OSError as err:
                msg = "%s: Cannot SIGINT the job `%s` with PID = %d\n" % (
                    str(err),
                    self.name,
                    pid,
                )
                print(msg)
                if self.logger is not None:
                    self.logger.error(msg)
        os.chdir(orig_dir)
        return 0
