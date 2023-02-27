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


def filename_fmt(*variants: str) -> str:
    variant = ".".join(variants)
    return f"data/D.{variant}.bin"


async def entry_preprocess():
    with open(FILENAME_RAW, "r", encoding="utf8") as file_input:
        dataset_raw = [rec for row in csv.reader(
            file_input) if (rec := Record.parse(row))]

    dataset_raw_stats = stats.of(dataset_raw)

    for filter, standardise, split in variants.datasets():
        dataset_clean: list[Record] = filter[1](
            dataset_raw, dataset_raw_stats)
        dataset_normal, encodings = standardise[1](
            dataset_clean, stats.of(dataset_clean))
        dataset_train, dataset_validate, dataset_test = split[1](
            dataset_normal)

        with open(filename_fmt(filter[0], standardise[0], split[0]), "wb") as file:
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
    print("[STDOUT]: {}\n[STDERR]: {}".format(*await msbuild.variant()))


async def entry_train():
    pass


async def main():
    try:
        await entry_preprocess()
        await entry_build()
        # await entry_train()
    except Exception:
        print_exc()

asyncio.run(main())
