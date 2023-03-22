import subprocess

import variants

import graph


def find_top(count: int) -> list[tuple[str, str, str]]:
    models: dict[float, tuple[str, str, str]] = {
        1000000: ("None", "None", "None")
    }
    for model in variants.train():
        name, dataset, _ = model
        if not dataset.startswith("std_dev_iqr_3.lin1-9.year_2_1_1"):
            continue
        data: list[tuple[float, ...]] = graph.read_log(name)
        best = min(data, key=lambda row: row[2])
        worst = max(sorted(list(models.keys())))
        if len(models) < count:
            models[best[2]] = (f"{model[0]}/{int(best[0])}", *model[1:])
        elif best[2] < worst:
            del models[worst]
            models[best[2]] = (f"{model[0]}/{int(best[0])}", *model[1:])
    return list(models.values())


def spawn(name: str, dataset: str, build: str):
    executable = f"bin/Release/train.{build}/mlp.exe"
    proc = subprocess.call([executable, name, dataset],
                           stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc
