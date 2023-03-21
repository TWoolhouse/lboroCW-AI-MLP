import csv
from itertools import cycle
from pathlib import Path
from typing import Collection

import matplotlib.pyplot as plt
import numpy
from matplotlib.axes import Axes
from record import FIELDS, Record
from standardise import Encoding

Path("graph/dataset").resolve().mkdir(parents=True, exist_ok=True)
Path("graph/model").resolve().mkdir(parents=True, exist_ok=True)
Path("graph/all").resolve().mkdir(parents=True, exist_ok=True)


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
        axes.plot([rec.date for rec in dataset], [getattr(rec, field) for rec in dataset],
                  ".", markersize=1, color=colour)
        axes.set_ylabel(field.replace("_", " ").title())

    print(f"\t{name}")
    fig.suptitle(name)
    fig.tight_layout(h_pad=0.5)
    fig.savefig(f"./graph/dataset/{name}.png", dpi=600)
    plt.close(fig)


def read_log(name: str) -> list[tuple[float, ...]]:
    def extract_log(row):
        try:
            return tuple(map(float, row))
        except ValueError:
            return None
    with open(f"model/training/{name}.log") as file:
        results = [row for r in csv.reader(
            file) if (row := extract_log(r)) is not None]
    return results


def model_training(name: str, name_ds: str, name_model: str):
    results = read_log(name)

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


def model_test_histogram(trainers: list[tuple[str, str, str]]):
    winners: list[tuple[tuple[str, str, str], tuple[float, ...]]] = []
    for name, dataset, model in trainers:
        results = read_log(name)
        best = ((name, dataset, model), min(results, key=lambda x: x[2]))
        winners.append(best)

    epoch_max = int(max(winners, key=lambda x: x[1][0])[1][0])
    non_epoch_max = len([None for m, v in winners if v[0] != epoch_max])
    print(
        f"\tNumber of Model not requiring the max {epoch_max} epochs: {non_epoch_max}")

    def histogram(ax: Axes):
        ax.hist([v[0] for m, v in winners], bins=20)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Frequency")
        ax.set_title(
            "Histogram of Epochs taken to Reach the Lowest Validation RMSE", loc="left")

    def scatter(ax: Axes):
        ax.plot(*list(zip(*[(v[0], v[2])
                for _, v in winners])), ".", markersize=4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("RMSE")
        ax.set_title("Epochs vs RMSE", loc="left")

    def pie(ax: Axes):
        ax.pie([non_epoch_max, len(winners) - non_epoch_max],
               explode=[0.1, 0.1],
               autopct="%d%%",
               radius=2,
               labels=[f"Best after <{epoch_max} epochs",
                       f"Best after {epoch_max} epochs"]
               )

    funcs = [scatter, pie]
    fig, axes = plt.subplots(len(funcs))
    if not isinstance(axes, numpy.ndarray):
        axes: list[Axes] = [axes]

    for ax, func in zip(axes, funcs):
        func(ax)

    fig.tight_layout(h_pad=1)
    fig.savefig("graph/all/best_epochs.png", dpi=400)
    plt.close(fig)
