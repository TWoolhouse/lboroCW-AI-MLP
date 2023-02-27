import subprocess


def spawn(name: str, dataset: str, build: str):
    executable = f"bin/Release/{build}/mlp.exe"
    proc = subprocess.call([executable, name, dataset],
                           stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc
