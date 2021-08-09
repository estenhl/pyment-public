import subprocess


def run(cmd: str, *, silence: bool = True):
    stdout = subprocess.DEVNULL if silence else None

    process = subprocess.Popen(cmd.split(' '), stdout=stdout)
    process.communicate()