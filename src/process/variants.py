import filter
import height
import split
import standardise


def all():
    return filter.VARIANTS, standardise.VARIANTS, split.VARIANTS


def datasets():
    return [(i, j, k) for i in filter.VARIANTS.items() for j in standardise.VARIANTS.items() for k in split.VARIANTS.items()]


def builds():
    # TODO: Get number of inputs
    return height.variants(5)
