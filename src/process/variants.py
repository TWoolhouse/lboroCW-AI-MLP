from typing import Optional

import activation
import filter as filter_
import height
import modification
import split
import standardise


def variant_name(*variants: Optional[str]) -> str:
    return ".".join(filter(None, variants))


def all():
    return filter_.VARIANTS, standardise.VARIANTS, split.VARIANTS


def datasets():
    return [(i, j, k) for i in filter_.VARIANTS.items() for j in standardise.VARIANTS.items() for k in split.VARIANTS.items()]


def builds():
    # TODO: Get number of inputs
    return [(i, j, k) for i in height.variants(5) for j in activation.VARIANTS.items() for k in modification.VARIANTS.items()]


def train():
    def vname(var):
        return variant_name(*(v[0] for v in var))
    return [(variant_name(vname(dataset), vname(build)), vname(dataset), vname(build)) for dataset in datasets() for build in builds()]
