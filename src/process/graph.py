import csv
from itertools import cycle
from pathlib import Path
from typing import Collection

import matplotlib.pyplot as plt
import numpy
from record import FIELDS, Record
from standardise import Encoding

Path("graph/dataset").resolve().mkdir(parents=True, exist_ok=True)
Path("graph/model").resolve().mkdir(parents=True, exist_ok=True)


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

    print(f"\t{name}")
    fig.tight_layout(h_pad=0.5)
    fig.savefig(f"./graph/dataset/{name}.png", dpi=600)
    plt.close(fig)


def model_training(name: str, name_ds: str, name_model: str):
    def extract(row):
        try:
            return tuple(map(float, row))
        except ValueError:
            return None

    with open(f"model/training/{name}.log") as file:
        results = [row for r in csv.reader(
            file) if (row := extract(r)) is not None]

    def standardise(data: Collection[float]) -> list[float]:
        a, b = min(data), max(data)
        rng = b - a
        return [(v - a) / rng for v in data]

    columns = tuple(zip(*results))
    epochs, error_train, error_validate, learning_rate = columns[:4]

    ds = [
        ("Training", error_train),
        ("Validation", error_validate),
        ("Learning Rate", learning_rate),
    ]

    def graph_error(ax):
        for (dname, data), colour in zip(ds[:2], cycle(["red", "blue", "purple"])):
            ax.plot(epochs, data, color=colour, label=dname.title())
            ax.set_xlabel("Epoch")
            ax.set_ylabel("RMSE")
            ax.set_title(f"{name_ds}\n{name_model}", pad=20, loc="left")

    def graph_learning_rate(ax):
        for (dname, data), colour in zip([ds[2]], cycle(["purple"])):
            ax.plot(epochs, data, color=colour)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Learning Rate")

    plotting = [func for func, pred in [
        (graph_error, lambda: True),
        (graph_learning_rate, lambda: any(
            name in name_model for name in ("bold_driver", "annealing"))),
    ] if pred()]

    fig, axes = plt.subplots(len(plotting))
    if not isinstance(axes, numpy.ndarray):
        axes = [axes]

    for ax, func in zip(axes, plotting):
        func(ax)

    fig.legend()
    fig.tight_layout(h_pad=0.5)
    fig.savefig(f"graph/model/{name}.png", dpi=400)
    plt.close(fig)
