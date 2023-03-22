from re import A
from typing import Literal, Optional

import activation
import filter as filter_
import height
import modification
import split
import standardise
from record import FIELDS


def variant_name(*variants: Optional[str]) -> str:
    return ".".join(filter(None, variants))


def datasets():
    return [(i, j, k) for i in filter_.VARIANTS.items() for j in standardise.VARIANTS.items() for k in split.VARIANTS.items()]


def builds():
    return [(i, j, k) for i in height.variants(len(FIELDS) - 1) for j in activation.VARIANTS.items() for k in modification.VARIANTS.items()]


def train():
    def vname(var):
        return variant_name(*(v[0] for v in var))
    return [(variant_name(vname(dataset), vname(build)), vname(dataset), vname(build)) for dataset in datasets() for build in builds()]


LABELS = [
    "filter",
    "standardise",
    "split",
    "height",
    "activation",
    "modification",
]

VARIANTS: dict[str, set[str]] = {}
TYPES: dict[Literal[
    "filter",
    "standardise",
    "split",
    "height",
    "activation",
    "modification",
], set[str]] = {}
ALL: set[str] = {n for n, _, _ in train()}

for name in ALL:
    for i, section in enumerate(name.split(".")):
        VARIANTS.setdefault(section, set()).add(name)
        TYPES.setdefault(LABELS[min(i, len(LABELS) - 1)], set()).add(section)
