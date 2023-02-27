import activation
import filter
import height
import split
import standardise


def variant_name(*variants: str) -> str:
    return ".".join(variants)


def all():
    return filter.VARIANTS, standardise.VARIANTS, split.VARIANTS


def datasets():
    return [(i, j, k) for i in filter.VARIANTS.items() for j in standardise.VARIANTS.items() for k in split.VARIANTS.items()]


def builds():
    # TODO: Get number of inputs
    return [(i, j) for i in height.variants(5) for j in activation.VARIANTS.items()]


def train():
    def vname(var):
        return variant_name(*(v[0] for v in var))
    return [(variant_name(vname(dataset), vname(build)), vname(dataset), vname(build)) for dataset in datasets() for build in builds()]
