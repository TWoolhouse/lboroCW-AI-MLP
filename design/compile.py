# pandoc report.md -o report.pdf --pdf-engine-opt=-shell-escape

import subprocess
import sys
from pathlib import Path

directory = Path.cwd()

markdown_files = [(f := directory / "design" / i, directory / "design" / "output" / f"{f.stem}.pdf")
                  for i in ["report.md", "code.md"]]


def files(*exts: str) -> list[Path]:
    src = directory / "src"
    files: list[Path] = []
    for ext in exts:
        for p in src.rglob(f"*{ext}"):
            if p.is_file():
                files.append(p)
    return files


def f2latex(files: list[Path]) -> str:
    return ", ".join(f".{str(i)[len(str(directory)):]}" for i in files).replace("\\", "/")


MAPPINGS = {
    "PYTHON_FILES": f2latex(files(".py")),
    "CPP_FILES": f2latex(files(".cpp", ".h")),
}

print(f"Compiling: {len(markdown_files)}")
for input, output in markdown_files:
    print(f"\t{input}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(input, "r") as file:
        content = file.read()
        for k, v in MAPPINGS.items():
            content = content.replace(f"PY_{k}", v)
    proc = subprocess.run(
        f'pandoc -o "{output}" --pdf-engine-opt=-shell-escape', input=content.encode("utf8"), capture_output=True, cwd=str(directory))

    try:
        proc.check_returncode()
    except Exception:
        print(f"{proc.stderr.decode('utf8')}")
        raise

print("Done!")
