import asyncio
import os
from typing import Literal

SOLUTION = "ai.sln"
EXECUTABLE = "msbuild"


async def trainer(variation: str, height: int, activation: str):
    return await compile(variation, {
        "TRAINING": None,
        "HEIGHT": height,
        f"ACTIVATION_{activation}": None
    })


async def compile(variation: str, defines: dict[str, str | int | None], configuration: Literal["Debug", "Release"] = "Release"):
    cmd = f"{EXECUTABLE} {SOLUTION} /m /verbosity:minimal /p:configuration={configuration}"

    env = {
        "MLP_VARIANT": variation,
        "MLP_BUILD_OPTIONS": "{defines}".format(
            defines=" ".join(
                f"/DMLP_{key.upper()}" if value is None else f"/DMLP_{key.upper()}={value}" for key, value in defines.items())
        ),
    }

    process = await asyncio.create_subprocess_shell(
        cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env={**env, **os.environ})
    stdout, stderr = await process.communicate()
    return process.returncode, stdout.decode("utf8"), stderr.decode("utf8")
