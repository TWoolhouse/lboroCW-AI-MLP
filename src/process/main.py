import argparse
import asyncio
import csv
import struct
from traceback import print_exc

import msbuild
import stats
import variants
from record import Record

FILENAME_RAW = "data/raw.csv"


def variant_name(*variants: str) -> str:
    return ".".join(variants)


def filename_fmt(variant: str) -> str:
    return f"data/D.{variant}.bin"


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
            f"\t{name: <50} {size} {size / len(dataset_raw) * 100:0>.2f}% - {len(dataset_train)}, {len(dataset_validate)}, {len(dataset_test)}")
        with open(filename_fmt(name), "wb") as file:
            file.write(struct.pack("@NNN", len(dataset_train),
                                   len(dataset_validate), len(dataset_test)))
            for encoding in encodings:
                file.write(encoding.serialise())
            for rec in dataset_train:
                file.write(rec.serialise())
            for rec in dataset_validate:
                file.write(rec.serialise())
            for rec in dataset_test:
                file.write(rec.serialise())


async def entry_build():
    # for variant in variants.
    _, stderr = await msbuild.variant()
    if stderr:
        print(f"[STDERR]: {stderr}")


async def entry_train():
    pass


async def main(args: argparse.Namespace):
    try:
        if (args.preprocess):
            await entry_preprocess()
        if (args.build):
            await entry_build()
        if (args.train):
            await entry_train()
    except Exception:
        print_exc()

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--preprocess", action="store_true",
                    help="Preprocess the raw dataset")
parser.add_argument("-b", "--build", action="store_true",
                    help="Build the training executables")
parser.add_argument("-t", "--train", action="store_true",
                    help="Train the models")

asyncio.run(main(parser.parse_args()))
