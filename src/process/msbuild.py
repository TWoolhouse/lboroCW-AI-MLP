import asyncio
import os
from typing import Literal

SOLUTION = "ai.sln"
EXECUTABLE = "msbuild"


async def variant(configuration: Literal["Debug", "Release"] = "Release"):
    cmd = f"{EXECUTABLE} {SOLUTION} /m /verbosity:minimal /p:configuration={configuration}"

    defines: dict[str, str | int | None] = {}

    env = {
        "MLP_VARIANT": "default",
        "MLP_BUILD_OPTIONS": "{defines}".format(
            defines=" ".join(
                f"/D{key}" if value is None else f"/D{key}={value}" for key, value in defines.items())
        ),
    }

    process = await asyncio.create_subprocess_shell(
        cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env={**env, **os.environ})
    stdout, stderr = await process.communicate()
    return stdout.decode("utf8"), stderr.decode("utf8")
