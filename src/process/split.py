import random
from typing import Callable, TypeAlias

from record import Record

Variant: TypeAlias = Callable[[list[Record]], tuple[
    list[Record], list[Record], list[Record]
]]


def variant_default(train: float, validation: float, test: float) -> Variant:
    def default(dataset: list[Record]) -> tuple[list[Record], list[Record], list[Record]]:
        size = len(dataset)
        train_set = dataset[:int(train * size)]
        validation_set = dataset[len(train_set):
                                 len(train_set) + int(validation * size)]
        test_set = dataset[len(train_set)+len(validation_set):]

        return train_set, validation_set, test_set
    return default


def variant_random(train: float, validation: float, test: float) -> Variant:
    def rnd(dataset: list[Record]) -> tuple[list[Record], list[Record], list[Record]]:
        dataset = dataset.copy()
        random.shuffle(dataset)
        size = len(dataset)
        train_set = dataset[:int(train * size)]
        validation_set = dataset[len(train_set):
                                 len(train_set) + int(validation * size)]
        test_set = dataset[len(train_set)+len(validation_set):]

        return train_set, validation_set, test_set
    return rnd


def variant_year(train: int, validation: int, test: int) -> Variant:
    def yr(dataset: list[Record]) -> tuple[list[Record], list[Record], list[Record]]:
        dataset = sorted(dataset, key=lambda rec: rec.date)
        years = sorted(list({rec.date.year for rec in dataset}))

        def extract(dataset: list[Record], years: list[int]) -> list[Record]:
            for index, record in enumerate(dataset):
                if record.date.year not in years:
                    break
            return dataset[:index]

        train_set = extract(dataset, years[:train])
        validation_set = extract(
            dataset[len(train_set):], years[train:train+validation])
        test_set = extract(
            dataset[len(train_set) + len(validation_set):], years[-test:])

        return train_set, validation_set, test_set

    return yr


VARIANTS: dict[str, Variant] = {
    # "default_60_20_20": variant_default(0.6, 0.2, 0.2),
    "random_60_20_20": variant_random(0.6, 0.2, 0.2),
    "year_2_1_1": variant_year(2, 1, 1),
}
