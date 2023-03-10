from itertools import cycle

import matplotlib.pyplot as plt
from record import FIELDS, Record
from standardise import Encoding


def processed(name: str, encodings: list[Encoding], train: list[Record], validate: list[Record], test: list[Record]):
    sets = (train, validate, test)
    for ds in sets:
        ds.sort(key=lambda rec: rec.date)
    ds = [*train, *validate, *test]
    ds.sort(key=lambda rec: rec.date)

    dataset(name, ds)
    dataset(f"{name}.raw", [Record(
        rec.date,
        encodings[0].decode(rec.temperature),
        encodings[1].decode(rec.wind_speed),
        encodings[2].decode(rec.solar_radiation),
        encodings[3].decode(rec.air_pressure),
        encodings[4].decode(rec.humidity),
        encodings[5].decode(rec.evaporation),
    ) for rec in ds])


COLOURS = ["red", "orange", "yellow", "green", "blue", "purple"]


def dataset(name: str, dataset: list[Record]):
    fig, ax = plt.subplots(len(FIELDS))
    fig.set_size_inches(10, 10)

    for (axes, field, colour) in zip(ax, FIELDS, cycle(COLOURS)):
        axes.plot([getattr(rec, field) for rec in dataset],
                  ".", markersize=1, color=colour)
        axes.set_ylabel(field.replace("_", " ").title())

    fig.savefig(f"./graph/dataset/{name}.png", dpi=100)
