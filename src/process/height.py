def values(inputs: int):
    # return [inputs, inputs * 2]
    return range(inputs, inputs * 2 + 1)


def variants(inputs: int) -> list[tuple[str, int]]:
    return [(f"H{h:0>2}", h) for h in values(inputs)]
