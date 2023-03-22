import argparse
import asyncio
import csv
import sys
import test
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from traceback import print_exc

import dataset
import msbuild
import numpy as np
import stats
import train
import variants
from dataset import filename_fmt
from record import FIELDS, Record
from standardise import Encoding
from variants import variant_name

import graph

FILENAME_RAW = "data/raw.csv"


async def entry_preprocess():
    print("Reading Raw Dataset:", FILENAME_RAW)
    with open(FILENAME_RAW, "r", encoding="utf8") as file_input:
        dataset_raw = [rec for row in csv.reader(
            file_input) if (rec := Record.parse(row))]

    dataset_raw_stats = stats.of(dataset_raw)
    print(f"\tFound {len(dataset_raw)} Records")

    vars = variants.datasets()
    print(f"Constructing Datasets: {len(vars)}")
    for filter, standardise, split in vars:
        dataset_clean: list[Record] = filter[1](
            dataset_raw, dataset_raw_stats)
        dataset_normal, encodings = standardise[1](
            dataset_clean, stats.of(dataset_clean))
        dataset_train, dataset_validate, dataset_test = split[1](
            dataset_normal)

        name = variant_name(filter[0], standardise[0], split[0])
        size = len(dataset_train) + len(dataset_validate) + len(dataset_test)
        print(
            f"\t{name: <50} {size} {size / len(dataset_raw) * 100:0>.2f}% - {len(dataset_train)} {len(dataset_train) / size * 100:0>.2f}%, {len(dataset_validate)} {len(dataset_validate) / size * 100:0>.2f}%, {len(dataset_test)} {len(dataset_test) / size * 100:0>.2f}%")
        dataset.serialise(name, encodings, dataset_train,
                          dataset_validate, dataset_test)


async def entry_build():
    build_jobs = [(name := variant_name(height[0], activation[0], mods[0]), msbuild.trainer(name, height[1], activation[1], mods[1]))
                  for height, activation, mods in variants.builds()]
    print(f"Compiling Trainers: {len(build_jobs)}")
    with ProcessPoolExecutor(max_workers=4) as pool:
        async with asyncio.TaskGroup() as tg:
            tasks = [(name, tg.create_task(asyncio.wait_for(asyncio.get_event_loop().run_in_executor(pool, job), timeout=None)))
                     for name, job in build_jobs]
            for build, task in tasks:
                task.add_done_callback(
                    partial(lambda name, value, *_: print(f"\t{value.result()[0]} - {name}"), build))
    for build, task in tasks:
        error_code, stdout, stderr = task.result()
        if error_code != 0 or stderr:
            raise RuntimeError(
                f"Failed to Compile: {build} with: {stdout}\n{stderr}")


async def entry_train():
    trainers = variants.train()
    print(f"Training: {len(trainers)}")
    with ProcessPoolExecutor(max_workers=4) as pool:
        jobs = [(name, asyncio.get_event_loop().run_in_executor(pool, partial(
            train.spawn, name, filename_fmt(dataset), build)))
            for name, dataset, build in trainers]
        for name, job in jobs:
            job.add_done_callback(
                partial(lambda name, value, *_: print(f"\t{value.result()} - {name}"), name))
        coros = [(name, asyncio.wait_for(job, timeout=None))
                 for name, job in jobs]
        async with asyncio.TaskGroup() as tg:
            for _, coro in coros:
                tg.create_task(coro)


async def entry_test():
    best: list[tuple[str, str, str]] = test.find_top(10)
    best_builds = {i[2] for i in best}

    build_jobs = [(name := ("train." + variant_name(height[0], activation[0], mods[0])), msbuild.tester(name, height[1], activation[1], mods[1]))
                  for height, activation, mods in variants.builds() if variant_name(height[0], activation[0], mods[0]) in best_builds]
    print(f"Compiling Testers: {len(build_jobs)}")
    with ProcessPoolExecutor(max_workers=4) as pool:
        async with asyncio.TaskGroup() as tg:
            tasks = [(name, tg.create_task(asyncio.wait_for(asyncio.get_event_loop().run_in_executor(pool, job), timeout=None)))
                     for name, job in build_jobs]
            for build, task in tasks:
                task.add_done_callback(
                    partial(lambda name, value, *_: print(f"\t{value.result()[0]} - {name}"), build))
    for build, task in tasks:
        error_code, stdout, stderr = task.result()
        if error_code != 0 or stderr:
            raise RuntimeError(
                f"Failed to Compile: {build} with: {stdout}\n{stderr}")

    print(f"Testing: {len(best)}")
    with ProcessPoolExecutor(max_workers=4) as pool:
        jobs = [(name, asyncio.get_event_loop().run_in_executor(pool, partial(
            test.spawn, name, filename_fmt(dataset), build)))
            for name, dataset, build in best]
        for name, job in jobs:
            job.add_done_callback(
                partial(lambda name, value, *_: print(f"\t{value.result()} - {name}"), name))
        coros = [(name, asyncio.wait_for(job, timeout=None))
                 for name, job in jobs]
        async with asyncio.TaskGroup() as tg:
            for _, coro in coros:
                tg.create_task(coro)


async def entry_analyse_dataset():
    graphs = variants.datasets()
    # 2 per dataset (std, as is) & 1 raw input
    print(f"Graphing Datasets: {len(graphs) * 2 + 1}")
    for (filter, _), (standardise, _), (split, _) in graphs:
        name = variant_name(filter, standardise, split)
        encodings, (train, validate, test) = dataset.deserialise(name)
        graph.processed(name, encodings, train, validate, test)


async def entry_analyse_model_single():
    trainers = variants.train()
    print(f"Graphing Model Training: {len(trainers)}")
    for name, dataset, build in trainers:
        print(f"\t{name}")
        graph.model_training(name, dataset, build)


async def entry_analyse_model_group():
    print(f"Graphing Groups: Dataset")
    graph.model_dataset()
    print(f"Graphing Groups: Modifications")
    graph.model_modifcations()


async def entry_analyse_model_best():
    trainers = variants.train()
    print("Graphing Epochs Taken")
    graph.model_epochs_taken(trainers)


async def entry_analyse_testing():
    best: list[tuple[str, str, str]] = test.find_top(10)

    print(f"Graphing Model Test Results: {len(best)}")
    for name, ds_name, _ in best:
        if name != "std_dev_iqr_3.lin1-9.year_2_1_1.H09.tanh.annealing/20000":
            continue
        with open(f"test/{name}.log", "r") as file:
            rmse: tuple[float, float, float] = tuple(
                map(float, file.readline().strip().split(",")))
            print(
                f"\t{name} | {rmse[0]:0>.5f} | {rmse[1]:0>.5f} | {rmse[2]:0>.5f}")
            ds: tuple[list[Encoding], tuple[list[Record],
                                            list[Record], list[Record]]] = dataset.deserialise(ds_name)
            encodings, sets = ds
            predictions: list[list[tuple[Record, float, float]]] = []
            for set in sets:
                pred: list[tuple[Record, float, float]] = []
                predictions.append(pred)
                for line, rec in zip(file, set):
                    a = tuple(map(float, line.strip().split(",")))
                    pred.append((rec, *a))
                errors = [guess - encodings[-1].decode(rec.evaporation)
                          for rec, _, guess in pred]
                # print(f"\t{sum(errors) / len(errors):0>.5f}", end=" | ")
                y = [rec.evaporation for rec, _, _ in pred]
                x_r: list[list[float]] = [[getattr(rec, field)
                                           for rec, _, _ in pred] for field in [f for f in FIELDS if f != "evaporation"]]

                x = np.transpose(x_r)
                x = np.c_[x, np.ones(x.shape[0])]  # add bias term
                linreg: list[float] = list(
                    np.linalg.lstsq(x, y, rcond=None)[0])
                linest = [sum([a * b for a, b in zip([*row, 1.0], linreg)])
                          for row in zip(*x_r)]
                raw = [encodings[-1].decode(rec.evaporation)
                       for rec, _, _ in pred]
                diff_linest = [g - r for g, r in zip(linest, raw)]
                print(
                    f"{sum([g - r for g, r in zip(linest, raw)]) / len(raw):0>.5f}", end=" | ")
                print(
                    f"{np.sqrt(sum([(g - r) ** 2 for g, r in zip(linest, raw)]) / len(raw)):0>.5f}",)
            print()
            graph.prediction(name, encodings[-1], predictions)


async def entry_report():
    # , stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    proc = await asyncio.subprocess.create_subprocess_exec(sys.executable, Path("design/compile.py").resolve())
    await proc.communicate()
    return proc.returncode


async def main(args: argparse.Namespace):
    try:
        if args.preprocess:
            await entry_preprocess()
        if args.analyse_dataset:
            await entry_analyse_dataset()
        if args.build:
            await entry_build()
        if args.train:
            await entry_train()
        if args.analyse_model_individual:
            await entry_analyse_model_single()
        if args.analyse_model_grouped:
            await entry_analyse_model_group()
        if args.analyse_model_best:
            await entry_analyse_model_best()
        if args.test:
            await entry_test()
        if args.analyse_testing:
            await entry_analyse_testing()
        if args.report:
            await entry_report()
    except Exception:
        print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--preprocess", action="store_true",
                        help="Preprocess the raw dataset")
    parser.add_argument("-b", "--build", action="store_true",
                        help="Build the training executables")
    parser.add_argument("-t", "--train", action="store_true",
                        help="Train the models")
    parser.add_argument("-ad", "--analyse-dataset", action="store_true",
                        help="Analyse the raw dataset and all generated datasets")
    parser.add_argument("-ami", "--analyse-model-individual", action="store_true",
                        help="Analyse the models validation data results")
    parser.add_argument("-amg", "--analyse-model-grouped", action="store_true",
                        help="Analyse the models validation data results relative to modifications")
    parser.add_argument("-amb", "--analyse-model-best", action="store_true",
                        help="Analyse the models validation data results compared to one another")
    parser.add_argument("-v", "--test", action="store_true",
                        help="Test the best models")
    parser.add_argument("-at", "--analyse-testing", action="store_true",
                        help="Analyse the models training results")
    parser.add_argument("-r", "--report", action="store_true",
                        help="Compile the Report")

    asyncio.run(main(parser.parse_args()))
