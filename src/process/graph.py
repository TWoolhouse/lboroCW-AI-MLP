import csv
from itertools import cycle
from pathlib import Path
from typing import Collection

import matplotlib.pyplot as plt
import numpy as np
import variants
from matplotlib.axes import Axes
from record import FIELDS, Record
from standardise import Encoding
from variants import variant_name

SUBDIRS = [
    "dataset",
    "model",
    "all",
    "best",
    "modification",
    "dataset_group",
]

for subdir in SUBDIRS:
    Path(f"graph/{subdir}").resolve().mkdir(parents=True, exist_ok=True)


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


COLOURS = ["red", "orange", "green", "blue", "purple", "black"]


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
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    for ax, func in zip(axes, plotting):
        func(ax)

    fig.legend()
    fig.tight_layout(h_pad=0.5)
    fig.savefig(f"graph/model/{name}.png", dpi=400)
    plt.close(fig)


def model_modifcations():
    prefixes = {".".join([v for i, v in enumerate(variant.split(".")) if i < (len(variants.TYPES) - 1)])
                for variant in variants.ALL
                }

    def graph_error(epochs, data, title, suffix, ax, colour):
        ax.plot(epochs, data, color=colour, label=suffix)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("RMSE")
        ax.set_title(title, loc="left")

    for prefix in prefixes:
        print(f"\t{prefix}")
        modded = {i for i in variants.ALL if i.startswith(prefix)}

        fig, (ax_train, ax_valid) = plt.subplots(2)

        for mod, colour in zip(modded, cycle(COLOURS)):
            data = read_log(mod)
            columns = tuple(zip(*data))
            epochs, error_train, error_validate, learning_rate = columns[:4]
            m_name = mod[len(prefix)+1:]
            if len(m_name.strip()) == 0:
                m_name = "default"
            graph_error(epochs, error_train, "Training",
                        m_name, ax_train, colour)
            graph_error(epochs, error_validate, "Validation",
                        None, ax_valid, colour)

        fig.suptitle(prefix)
        fig.legend(ncols=2, loc="upper right", bbox_to_anchor=(1.0, 0.95))
        fig.tight_layout(h_pad=0.5)
        fig.savefig(f"graph/modification/{prefix}.png")
        plt.close(fig)
    # results = read_log(name)


def model_dataset():
    builds = {build for _, _, build in variants.train()}

    def graph_error(epochs, data, title, suffix, ax, colour):
        ax.plot(epochs, data, color=colour, label=suffix)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("RMSE")
        ax.set_title(title, loc="left")

    for build in builds:
        models = variants.ALL.copy()
        for v in build.split("."):
            models.intersection_update(variants.VARIANTS[v])
        models = {i for i in models if i.endswith(build)}
        print(f"\t{build}")

        fig, (ax_train, ax_valid) = plt.subplots(2)

        for model, colour in zip(models, cycle(COLOURS)):
            data = read_log(model)
            columns = tuple(zip(*data))
            epochs, error_train, error_validate, learning_rate = columns[:4]
            d_name = model[:-(len(build) + 1)]
            graph_error(epochs, error_train, "Training",
                        d_name, ax_train, colour)
            graph_error(epochs, error_validate, "Validation",
                        None, ax_valid, colour)

        fig.suptitle(build)
        fig.legend(ncols=1, loc="upper right", bbox_to_anchor=(1.0, 0.95))
        fig.tight_layout(h_pad=0.5)
        fig.savefig(f"graph/dataset_group/{build}.png")
        plt.close(fig)


def model_epochs_taken(trainers: list[tuple[str, str, str]]):
    winners: list[tuple[tuple[str, str, str], tuple[float, ...]]] = []
    for name, dataset, model in trainers:
        results = read_log(name)
        best = ((name, dataset, model), min(results, key=lambda x: x[2]))
        winners.append(best)

    epoch_max = int(max(winners, key=lambda x: x[1][0])[1][0])
    non_epoch_max = len([None for m, v in winners if v[0] != epoch_max])
    print(
        f"\tNumber of Models not requiring the max {epoch_max} epochs: {non_epoch_max}")

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
               autopct="%.2f%%",
               radius=2,
               labels=[f"Best after <{epoch_max} epochs",
                       f"Best after {epoch_max} epochs"]
               )

    funcs = [scatter, pie]
    fig, axes = plt.subplots(len(funcs))
    if not isinstance(axes, np.ndarray):
        axes: list[Axes] = [axes]

    for ax, func in zip(axes, funcs):
        func(ax)

    fig.tight_layout(h_pad=1)
    fig.savefig("graph/all/best_epochs.png", dpi=400)
    plt.close(fig)


def prediction(name: str, encoding: Encoding, predictions: list[list[tuple[Record, float, float]]]):
    flat = [r for set in predictions for r in set]
    preds = sorted(flat, key=lambda x: x[0].date)

    data = [(
            rec.date,
            rec.evaporation,
            guess,
            encoding.decode(rec.evaporation),
            val
            )
            for rec, guess, val in preds
            ]

    dates, evap, guess, evap_raw, guess_raw = zip(*data)

    fig, (ax_scatter, ax_line) = plt.subplots(2)
    fig.set_size_inches(32 / 9 * 4, 2 * 4)

    for (d, n), colour in zip([(guess_raw, "Prediction"), (evap_raw, "Dataset")], cycle(["red", "blue"])):
        ax_scatter.plot(dates, d, "o", markersize=2, label=n, color=colour)
    ax_scatter.set_title("Pan Evaporation on a Specific Date")
    ax_scatter.set_ylabel("Pan Evaporation")

    diff = [g - r for g, r in zip(guess_raw, evap_raw)]
    for (d, n), colour in zip([(diff, None)], cycle(["red", "blue"])):
        ax_line.plot(dates, d, "o", markersize=2, label=n, color=colour)
    ax_line.set_title("Difference between the Actual Value and the Prediction")
    ax_line.set_xlabel("Date")
    ax_line.set_ylabel("Error")

    nm = name.replace("/", ".")
    fig.suptitle(name)
    fig.legend()
    fig.savefig(f"graph/best/{nm}.png", dpi=500)
    plt.close(fig)

    # LINEAR REGRESSION

    y = [rec.evaporation for rec, _, _ in preds]
    x_r: list[list[float]] = [[getattr(rec, field)
                               for rec, _, _ in preds] for field in [f for f in FIELDS if f != "evaporation"]]

    x = np.transpose(x_r)
    x = np.c_[x, np.ones(x.shape[0])]  # add bias term
    linreg: list[float] = list(np.linalg.lstsq(x, y, rcond=None)[0])

    fig, (ax_scatter, ax_line) = plt.subplots(2)
    fig.set_size_inches(32 / 9 * 4, 2 * 4)

    linest = [sum([a * b for a, b in zip([*row, 1.0], linreg)])
              for row in zip(*x_r)]

    for (d, n), colour in zip([(guess_raw, "Prediction"), (evap_raw, "Dataset"), (linest, "Linear Regression")], cycle(["red", "purple", "blue"])):
        ax_scatter.plot(dates, d, "o", markersize=2, label=n, color=colour)
    ax_scatter.set_title("Pan Evaporation on a Specific Date")
    ax_scatter.set_ylabel("Pan Evaporation")

    diff_pred = [g - r for g, r in zip(guess_raw, evap_raw)]
    diff_linest = [g - r for g, r in zip(linest, evap_raw)]
    for (d, n), colour in zip([(diff_pred, None), (diff_linest, None)], cycle(["red", "blue"])):
        ax_line.plot(dates, d, "o", markersize=2, label=n, color=colour)
    ax_line.set_title("Difference between the Actual Value and the Prediction")
    ax_line.set_xlabel("Date")
    ax_line.set_ylabel("Error")

    nm = name.replace("/", ".")
    fig.suptitle(name)
    fig.legend()
    fig.savefig(f"graph/best/{nm}.linreg.png", dpi=500)
    plt.close(fig)
