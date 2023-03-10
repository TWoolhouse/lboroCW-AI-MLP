import struct

from record import Record
from standardise import Encoding


def filename_fmt(variant: str) -> str:
    return f"data/D.{variant}.bin"


fmt_size = struct.Struct("@NNNN")


def serialise(name: str, encodings: list[Encoding], train: list[Record], validate: list[Record], test: list[Record]):
    with open(filename_fmt(name), "wb") as file:
        file.write(fmt_size.pack(len(train),
                   len(validate), len(test), len(encodings)))
        for encoding in encodings:
            file.write(encoding.serialise())
        for rec in train:
            file.write(rec.serialise())
        for rec in validate:
            file.write(rec.serialise())
        for rec in test:
            file.write(rec.serialise())


def deserialise(name: str) -> tuple[list[Encoding], tuple[list[Record], list[Record], list[Record]]]:
    with open(filename_fmt(name), "rb") as file:
        size_train, size_validate, size_test, size_encodings = fmt_size.unpack_from(
            file.read(fmt_size.size))

        encodings = [Encoding.deserialise(file) for _ in range(size_encodings)]
        train = [Record.deserialise(file) for _ in range(size_train)]
        validate = [Record.deserialise(file) for _ in range(size_validate)]
        test = [Record.deserialise(file) for _ in range(size_test)]

        return (encodings, (train, validate, test))
