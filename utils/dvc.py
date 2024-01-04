import subprocess
from functools import partial


class DVCManager:
    def __init__(self):
        self.cmd = "dvc {} {}"
        self.run = partial(subprocess.run, shell=True, check=True)

    def add(self, path: str):
        cmd = self.cmd.format("add", path)
        self.run(cmd)

    def pull(self, path: str):
        cmd = self.cmd.format("pull", path)
        self.run(cmd)
