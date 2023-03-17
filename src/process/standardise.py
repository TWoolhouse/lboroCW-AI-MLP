import struct
from dataclasses import dataclass
from typing import Callable, Self, TypeAlias

from record import Record
from stats import Statistics


@dataclass(frozen=True)
class Encoding:
    min: float
    max: float
    lower: float
    upper: float

    def encode(self, x: float) -> float:
        norm = (x - self.min) / (self.max - self.min)
        return norm * (self.upper - self.lower) + self.lower

    def decode(self, x: float) -> float:
        norm = (x - self.lower) / (self.upper - self.lower)
        return norm * (self.max - self.min) + self.min

    @staticmethod
    def fmt():
        return struct.Struct("@dddd")

    def serialise(self) -> bytes:
        return self.fmt().pack(self.min, self.max, self.lower, self.upper)

    @classmethod
    def deserialise(cls, file) -> Self:
        return cls(*cls.fmt().unpack_from(file.read(cls.fmt().size)))


Variant: TypeAlias = Callable[[
    list[Record], Statistics], tuple[list[Record], list[Encoding]]]


def variant_linear(lower: float, upper: float) -> Variant:
    rng = upper - lower

    def std(x, stat):
        stat = stat[1]
        norm = (x - stat.min) / (stat.max - stat.min)
        return norm * rng + lower

    def linear(dataset: list[Record], stats: Statistics) -> tuple[list[Record], list[Encoding]]:
        encodings: list[Encoding] = [
            Encoding(stat.min, stat.max, lower, upper) for field, stat in stats]

        return [Record(
            record.date,
            std(record.temperature, stats[0]),
            std(record.wind_speed, stats[1]),
            std(record.solar_radiation, stats[2]),
            std(record.air_pressure, stats[3]),
            std(record.humidity, stats[4]),
            std(record.evaporation, stats[5]),
        ) for record in dataset], encodings
    return linear


VARIANTS: dict[str, Variant] = {
    "lin1-9": variant_linear(0.1, 0.9)
}
