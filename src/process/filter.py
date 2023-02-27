from typing import Callable, TypeAlias
from record import Record
from stats import Statistics

Variant: TypeAlias = Callable[[list[Record], Statistics], list[Record]]


def variant_standard_deviation(deviations: int) -> Variant:
    def std_dev(dataset: list[Record], stats: Statistics) -> list[Record]:
        new: list[Record] = []
        for record in dataset:
            for (field, stat) in stats:
                val = getattr(record, field)
                mean = stat.mean
                dev = (deviations * stat.deviation)
                if (val < (mean - dev)) or ((mean + dev) < val):
                    break
            else:
                new.append(record)
        return new
    return std_dev


def variant_inter_quartile_range(deviations: int) -> Variant:
    def iqr(dataset: list[Record], stats: Statistics) -> list[Record]:
        new: list[Record] = []
        for record in dataset:
            for (field, stat) in stats:
                val = getattr(record, field)
                median = stat.quartile[1]
                iqr = (deviations * stat.quartile_range)
                if (val < (median - iqr)) or ((median + iqr) < val):
                    break
            else:
                new.append(record)
        return new
    return iqr


VARIANTS: dict[str, Variant] = {
    "std_dev_3": variant_standard_deviation(3),
    "iqr_3": variant_standard_deviation(3),
}
