import numpy as np
import cv2 as cv
from utils import *


def test():
    for R in range(5, 10):
        for C in range(5, 10):
            out_image = generate_chess_board(cell_size=100, rows=R, cols=C)
            cv.imshow('Board', out_image)
            cv.waitKey(50)


def generate_chess_board(cell_size: int, rows: int, cols: int, pad: int = 30) -> np.ndarray:
    """

    :param cell_size: [int]
    :param rows:
    :param cols:
    :param pad:
    :return:
    """
    # Check inputs:
    assert isinstance(cell_size, int), TypeError(f"cell_size have to be int. got {type(cell_size)}.")
    assert cell_size > 0, ValueError(f"cell_size have to be positive. got {cell_size}.")

    assert isinstance(rows, int), TypeError(f"rows have to be int. got {type(rows)}.")
    assert rows > 0, ValueError(f"rows have to be positive. got {rows}.")

    assert isinstance(cols, int), TypeError(f"cols have to be int. got {type(cols)}.")
    assert cols > 0, ValueError(f"cols have to be positive. got {cols}.")

    assert isinstance(pad, int), TypeError(f"pad have to be int. got {type(pad)}.")
    assert pad >= 0, ValueError(f"pad have to be positive. got {pad}.")

    # Create empty Black image
    board_image: np.array = 255 * np.ones((cell_size * rows + 2 * pad, cell_size * cols + 2 * pad), dtype=np.uint8)

    # Number of squares:
    n_squares: int = rows * cols

    for i in range(n_squares):
        # 1.Rectangle Points:
        # 1.1 Top Left Rectangle point
        p1 = (cell_size * (i % cols) + pad, cell_size * (i // cols) + pad)

        # 1.2 Bottom Right Rectangle point
        p2 = (p1[0] + cell_size, p1[1] + cell_size)

        # 2.Rectangle Color(Black or White) according to position:
        # Where (a,b) in cell location in matrix
        # a = (i // cols) % 2  # even(0) or odd(1) row
        # b = (i % cols) % 2  # even(1) or odd(1) col
        # Color = 0 if (a == b) else 1  # 0-black, 1-white
        color = 255 * ((i // cols) % 2 != (i % cols) % 2)

        board_image = cv.rectangle(img=board_image,
                                   pt1=p1,
                                   pt2=p2,
                                   color=(color, color, color),
                                   thickness=-1)

    return board_image


def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Argument Parser Example")

    # Add arguments
    parser.add_argument("-c", "--columns", type=validate_positive, default=16, help="Number of Columns positive ")
    parser.add_argument("-r", "--rows", type=validate_positive, default=12, help="Number of Rows ")
    parser.add_argument("-cs", "--cell_size", type=validate_non_negative, default=40, help="Square cell size")
    parser.add_argument("-p", "--padding", type=validate_positive, required=False, default=50, help="A boolean input")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Perform actions based on the input arguments
    print(f"Creating Board {args.rows}x{args.columns}, Square size {args.cell_size}. and padding {args.padding}")
    return args


if __name__ == "__main__":

    # Call the main function with the parsed arguments
    args = main()

    chessboard_image = generate_chess_board(cell_size=args.cell_size, rows=args.rows, cols=args.columns)

    cv.imshow("Board",chessboard_image)
    cv.waitKey(0)
