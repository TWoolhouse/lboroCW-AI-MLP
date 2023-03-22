import asyncio
import os
import subprocess
from functools import partial
from typing import Literal

SOLUTION = "ai.sln"
EXECUTABLE = "msbuild"


def trainer(variation: str, height: int, activation: str, mods: list[tuple[str, str | int | float | None]]):
    return compile(variation, {
        "TRAINING": None,
        "HEIGHT": height,
        f"ACTIVATION_{activation}": None,
        **{
            f"TRAIN_{k.upper()}": v for k, v in mods if k is not None
        }
    })


def tester(variation: str, height: int, activation: str, mods: list[tuple[str, str | int | float | None]]):
    return compile(variation, {
        "HEIGHT": height,
        f"ACTIVATION_{activation}": None,
        **{
            f"TRAIN_{k.upper()}": v for k, v in mods if k is not None
        }
    })


def subproc(*args, env={}, capture_output=None, **kwargs) -> tuple[int, str, str]:
    proc = subprocess.run(*args, capture_output=True,
                          env={**env, **os.environ}, **kwargs)
    return proc.returncode, proc.stdout.decode("utf8"), proc.stderr.decode("utf8")


def compile(variation: str, defines: dict[str, str | int | float | None], configuration: Literal["Debug", "Release"] = "Release"):
    cmd = f"{EXECUTABLE} {SOLUTION} /m /verbosity:minimal /p:configuration={configuration}"

    env = {
        "MLP_VARIANT": variation,
        "MLP_BUILD_OPTIONS": "{defines}".format(
            defines=" ".join(
                f"/DMLP_{key.upper()}" if value is None else f"/DMLP_{key.upper()}=\"{value}\"" for key, value in defines.items())
        ),
    }

    return partial(subproc, cmd, env=env)
