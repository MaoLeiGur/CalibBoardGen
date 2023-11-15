import argparse


def validate_non_negative(value):
    try:
        int_value = int(value)
        if int_value >= 0:
            return int_value
        else:
            raise ValueError("Input must be greater than or equal to zero")
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid non-negative integer value")


def validate_positive(value):
    try:
        int_value = int(value)
        if int_value > 0:
            return int_value
        else:
            raise ValueError("Input must be greater than or equal to zero")
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid non-negative integer value")
