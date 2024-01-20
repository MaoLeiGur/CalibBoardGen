import numpy as np
import cv2 as cv
from utils import *
import time
from termcolor import colored

# import albumentations as A


# tags_families_names = {d: getattr(cv.aruco, d) for d in dir(cv.aruco) if d.startswith("DICT")}
TAGS_FAMILIES = {d: getattr(cv.aruco, d) for d in dir(cv.aruco) if d.startswith("DICT")}


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chessboard Generator:")

    # Add arguments
    parser.add_argument("-c", "--columns",
                        type=arg_positive_int,
                        default=10,
                        help="Number of Columns positive ")
    parser.add_argument("-r", "--rows",
                        type=arg_positive_int,
                        default=5,
                        help="Number of Rows ")
    parser.add_argument("-s", "--cell_size",
                        type=arg_positive_int,
                        default=50,
                        help="Square cell size")
    parser.add_argument("-p", "--padding",
                        type=arg_positive_int,
                        required=False,
                        default=50, help="A boolean input")

    # Boolean flag
    parser.add_argument("--viz",
                        action="store_true",
                        help="Enable verbose mode.")

    # Boolean flag
    parser.add_argument("--save",
                        default=False,
                        action="store_true",
                        help="Enable saving model to file.")

    # Parse the command-line arguments
    _args = parser.parse_args()
    # Perform actions based on the input arguments
    print(f"Creating Board {_args.rows}x{_args.columns}, Square size {_args.cell_size}. and padding {_args.padding}")

    # Example of using the parsed arguments
    for name, value in vars(_args).items():
        print(name, colored(f"{value}", "green"))

    return _args


def format_time(elapsed_time_ns: int) -> str:
    """
    gets an int numter that represent time on ns. and return string by time scale.
    :param elapsed_time_ns:
    :return:
    """
    if elapsed_time_ns >= 1e9:  # 1 second = 1e9 nanoseconds
        return f"{elapsed_time_ns / 1e9:.2f} [s]"
    elif elapsed_time_ns >= 1e6:  # 1 millisecond = 1e6 nanoseconds
        return f"{elapsed_time_ns / 1e6:.2f} [ms]"
    elif elapsed_time_ns >= 1e3:  # 1 microsecond = 1e3 nanoseconds
        return f"{elapsed_time_ns / 1e3:.2f} [us]"
    else:
        return f"{elapsed_time_ns} [ns]"


def test():
    for R in range(5, 10):
        for C in range(5, 10):
            out_image = generate_chess_board(size=100, rows=R, cols=C)
            cv.imshow('Board', out_image)
            cv.waitKey(50)


def generate_chess_board(size: int, rows: int, cols: int, pad: int = 30) -> np.ndarray:
    """

    :param size: [int]
    :param rows:
    :param cols:
    :param pad:
    :return:
    """
    # Check inputs:
    assert isinstance(size, int), TypeError(f"cell_size have to be int. got {type(size)}.")
    assert size > 0, ValueError(f"cell_size have to be positive. got {size}.")

    assert isinstance(rows, int), TypeError(f"rows have to be int. got {type(rows)}.")
    assert rows > 0, ValueError(f"rows have to be positive. got {rows}.")

    assert isinstance(cols, int), TypeError(f"cols have to be int. got {type(cols)}.")
    assert cols > 0, ValueError(f"cols have to be positive. got {cols}.")

    assert isinstance(pad, int), TypeError(f"pad have to be int. got {type(pad)}.")
    assert pad >= 0, ValueError(f"pad have to be positive. got {pad}.")

    # Create empty Black image
    board_image: np.array = 255 * np.ones((size * rows + 2 * pad, size * cols + 2 * pad), dtype=np.uint8)

    # Number of squares:
    n_squares: int = rows * cols

    for i in range(n_squares):
        # 1.Rectangle Points:
        # 1.1 Top Left Rectangle point
        p1 = (size * (i % cols) + pad, size * (i // cols) + pad)

        # 1.2 Bottom Right Rectangle point
        p2 = (p1[0] + size, p1[1] + size)

        # 2.Rectangle Color(Black or White) according to index-position [a,b]
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


def generate_tag_board(size: int, rows: int, cols: int, pad: int = 30, family: str = '') -> np.ndarray:
    """

    :param size:
    :param size: [int]
    :param rows:
    :param cols:
    :param pad:
    :return:
    """
    global TAGS_FAMILIES
    # Create empty Black image
    board_image: np.array = 255 * np.ones((size * rows + 2 * pad, size * cols + 2 * pad), dtype=np.uint8)

    # Number of squares:
    n_squares: int = rows * cols
    tag_ref: int = round(0.1 * size)
    tag_size: int = round(0.8 * size)

    tag_dict = cv.aruco.getPredefinedDictionary(TAGS_FAMILIES[family])
    n_tags_in_family = tag_dict.bytesList.shape[0]

    if n_squares > n_tags_in_family:
        raise ValueError(f"Too many tag requested in {family}. {n_tags_in_family}<{n_squares}")

    tag_id: int = 0
    for i in range(n_squares):
        # 1.Rectangle Points:
        # 1.1 Top Left Rectangle point
        p1 = (size * (i % cols) + pad, size * (i // cols) + pad)

        # 1.2 Bottom Right Rectangle point
        p2 = (p1[0] + size, p1[1] + size)

        # 2.Rectangle Color(Black or White) according to index-position [a,b]
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

        if color > 0:
            tag_image = tag_dict.generateImageMarker(id=tag_id, sidePixels=tag_size)
            board_image[p1[1] + tag_ref:p1[1] + tag_ref + tag_size,
            p1[0] + tag_ref:p1[0] + tag_ref + tag_size] = tag_image
            tag_id += 1

    return board_image


def main():
    # Create an ArgumentParser object

    return None


if __name__ == "__main__":
    # Call the main function with the parsed arguments

    args = parse_args()
    tag_family: str = 'DICT_APRILTAG_36h11'

    for dk in TAGS_FAMILIES.keys():
        t1 = time.time()
        chessboard_image = generate_tag_board(size=args.cell_size, rows=args.rows, cols=args.columns, family=dk)
        t2 = time.time()
        print(dk, 'Process Time: ', t2 - t1)

        tag_dict = cv.aruco.getPredefinedDictionary(TAGS_FAMILIES[dk])

        height, width = chessboard_image.shape
        board_image = np.zeros((height, width, 3), dtype=np.uint8)

        # Copy the grayscale data to all three channels
        board_image[:, :, 0] = chessboard_image  # Red channel
        board_image[:, :, 1] = chessboard_image  # Green channel
        board_image[:, :, 2] = chessboard_image  # Blue channel
        corners, ids, _ = cv.aruco.detectMarkers(board_image, tag_dict)
        board_image = cv.aruco.drawDetectedMarkers(board_image, corners, ids)
        cv.imshow("Board", board_image)
        cv.waitKey(0)
