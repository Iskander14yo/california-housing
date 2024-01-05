import subprocess
from functools import partial
from pathlib import Path


class DVCManager:
    def __init__(self):
        self.cmd = "dvc {} {}"
        self.run = partial(subprocess.run, shell=True, check=True)

    def add(self, path: str | Path):
        cmd = self.cmd.format("add", path)
        self.run(cmd)

    def pull(self, path: str | Path):
        cmd = self.cmd.format("pull", path)
        self.run(cmd)
