import dataclasses
from math import sqrt
from typing import Collection, TypeAlias

from record import Record, FIELDS


@dataclasses.dataclass(frozen=True, slots=True)
class Stat:
    size: int
    min: float
    max: float
    mean: float
    deviation: float
    quartile: tuple[float, float, float]
    quartile_range: float


Statistics: TypeAlias = list[tuple[str, Stat]]


def of(dataset: list[Record]) -> Statistics:
    fields: dict[str, list[float]] = {
        field: [getattr(rec, field) for i, rec in enumerate(dataset)] for field in FIELDS}

    stats: Statistics = []
    for field, column in fields.items():
        column.sort()
        q1 = quartile(column, 0.25)
        q3 = quartile(column, 0.75)
        stats.append((field, Stat(
            len(dataset),
            min(column),
            max(column),
            mean(column),
            standard_deviation(column),
            (
                q1,
                quartile(column, 0.5),
                q3,
            ),
            q3 - q1)
        ))
    return stats


def mean(collection: Collection[float]) -> float:
    return sum(collection) / len(collection)


def standard_deviation(collection: Collection[float]) -> float:
    avg = mean(collection)
    variance = sum((i - avg)**2 for i in collection) / len(collection)
    return sqrt(variance)


def quartile(collection: Collection[float], sector: float) -> float:
    assert 0 <= sector <= 1, "Sector must be in the range [0, 1]"
    return collection[int(sector * len(collection))]
