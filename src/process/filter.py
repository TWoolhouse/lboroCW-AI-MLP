from typing import Callable, TypeAlias

from record import Record
from stats import Statistics

Variant: TypeAlias = Callable[[list[Record], Statistics], list[Record]]


def variant_identity() -> Variant:
    def identity(dataset: list[Record], stats: Statistics) -> list[Record]:
        return dataset
    return identity


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


def variant_standard_deviation_inter_quartile_range(deviations: int) -> Variant:
    std_dev = variant_standard_deviation(deviations)
    iqr = variant_inter_quartile_range(deviations)

    def compose(dataset: list[Record], stats: Statistics) -> list[Record]:
        return iqr(std_dev(dataset, stats), stats)
    return compose


VARIANTS: dict[str, Variant] = {
    # "identity": variant_identity(),
    # "std_dev_3": variant_standard_deviation(3),
    # "std_dev_2": variant_standard_deviation(2),
    # "std_dev_1": variant_standard_deviation(1),
    # "iqr_3": variant_inter_quartile_range(3),
    # "iqr_2": variant_inter_quartile_range(2),
    # "iqr_1": variant_inter_quartile_range(1),
    "std_dev_iqr_3": variant_standard_deviation_inter_quartile_range(3),
    # "std_dev_iqr_2": variant_standard_deviation_inter_quartile_range(2),
    # "std_dev_iqr_1": variant_standard_deviation_inter_quartile_range(1),
}
