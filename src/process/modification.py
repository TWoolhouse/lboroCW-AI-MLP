import itertools
from typing import Any, Optional

variants = [
    ("momentum", 0.9),
    [("bold_driver", (0.001, 0.5, 1.01, 0.97)), ("annealing", (0.1, 0.01))],
    # "weight_decay"
]


def opt_to_input(opt: str | tuple[str, Any]) -> tuple[str, Any]:
    if isinstance(opt, tuple):
        return [(opt[0], v) for v in opt[1:]]
    return [(opt, None)]


options = [
    [(None, None), *itertools.chain.from_iterable(map(opt_to_input, opt))] if isinstance(opt, list) else [(None, None), *opt_to_input(opt)] for opt in variants
]


def vname(names: list[Optional[str]]) -> str:
    return ".".join(filter(None, names))


VARIANTS: dict[str, list[tuple[str, Optional[str | int | float]]]] = {}
for opt in itertools.product(*options):
    VARIANTS[vname(tuple(zip(*opt))[0])] = opt
