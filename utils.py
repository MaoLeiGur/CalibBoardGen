import argparse


def arg_positive_int(value: str) -> int:
    """

    :param value: string
    :return: cast value from string to int.
    """

    if not isinstance(value, str):
        raise TypeError(f"Wrong Type, got {type(value)} instead of str")

    int_value = int(value)

    if not int_value > 0:
        raise argparse.ArgumentTypeError(f"{value} is an invalid positive int value")

    return int_value
