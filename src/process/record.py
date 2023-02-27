from dataclasses import dataclass
import dataclasses
import datetime
from typing import Self
import struct


FIELDS = ["temperature", "wind_speed",
          "solar_radiation", "dsp", "drh", "evaporation"]


@dataclass(eq=True, frozen=True, slots=True)
class Record:
    date: datetime.date
    temperature: float
    wind_speed: float
    solar_radiation: float
    air_pressure: float
    humidity: float
    evaporation: float

    @classmethod
    def parse(cls, record: list[str]) -> Self | None:
        date = record[0].replace(",", "")
        while (len(date) < 6):
            date = "0" + date
        try:
            return cls(
                datetime.datetime.strptime(date, "%m%d%y").date(),
                float(record[1]),
                float(record[2]),
                float(record[3]),
                float(record[4]),
                float(record[5]),
                float(record[6]),
            )
        except ValueError:
            return

    def serialise(self) -> bytes:
        # uint16, uint8 * 2, double * 6
        return struct.pack("@HBBdddddd",
                           self.date.year, self.date.month, self.date.day,
                           self.temperature, self.wind_speed, self.solar_radiation, self.air_pressure, self.humidity, self.evaporation
                           )


FIELDS = [field.name for field in dataclasses.fields(
    Record) if field.type == float]
