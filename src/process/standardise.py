from dataclasses import dataclass
import struct
from typing import Callable, TypeAlias
from stats import Statistics
from record import Record


@dataclass(frozen=True)
class Encoding:
    min: float
    max: float
    lower: float
    upper: float

    def serialise(self) -> bytes:
        return struct.pack("@dddd", self.min, self.max, self.lower, self.upper)


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
    "linear_1-9_e-1": variant_linear(0.1, 0.9)
}
