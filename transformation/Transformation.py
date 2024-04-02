import argparse
import numpy
from plantcv import plantcv as pcv
from typing import List
from dataclasses import dataclass


@dataclass
class PcvImage:
    img: numpy.ndarray
    path: str
    img_name: str


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Image transformation.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "filename",
        type=str,
        nargs="?",
        default="",
        help="Specify the path of the image file.",
    )
    group.add_argument(
        "-src",
        type=str,
        dest="src",
        help="Specify the source directory.",
    )
    parser.add_argument(
        "-dst",
        type=str,
        help="Specify the destination directory.",
        default="./data/processed/",
    )
    parser.add_argument(
        "-blur",
        type=bool,
        action=argparse.BooleanOptionalAction,
        dest="blur",
        help="Apply gaussian blur processing.",
        default=False,
    )
    parser.add_argument(
        "-mask",
        type=bool,
        action=argparse.BooleanOptionalAction,
        dest="mask",
        help="Apply mask processing.",
        default=False,
    )
    parser.add_argument(
        "-roi",
        type=bool,
        action=argparse.BooleanOptionalAction,
        dest="roi",
        help="Apply roi objects processing.",
        default=False,
    )
    parser.add_argument(
        "-analyse",
        type=bool,
        action=argparse.BooleanOptionalAction,
        dest="analyse",
        help="Apply analyse object processing.",
        default=False,
    )
    parser.add_argument(
        "-pseudolandmarks",
        type=bool,
        action=argparse.BooleanOptionalAction,
        dest="pseudolandmarks",
        help="Apply pseudolandmarks processing.",
        default=False,
    )
    parser.add_argument(
        "-color",
        type=bool,
        action=argparse.BooleanOptionalAction,
        dest="color",
        help="Apply color histogram processing.",
        default=False,
    )
    args = parser.parse_args()
    return args


def read_images(src: List) -> list:
    images = []
    for img in src:
        img, path, img_name = pcv.readimage(filename=img)
        images.append(PcvImage(img=img, path=path, img_name=img_name))
    return images


def main():
    src_files = []
    args = parse_arguments()
    filename = args.filename
    src = args.src
    print("Selected flags:")
    print("Mask processing:", args.mask)
    print("ROI objects processing:", args.roi)
    print("Analyse object processing:", args.analyse)
    print("Pseudolandmarks processing:", args.pseudolandmarks)
    print("Color histogram processing:", args.color)
    print("Applying image transformation...")
    if filename:
        print("Processing single image...", filename)
        src_files.append(filename)
    if src:
        print("Source directory specified.", src)
        # Read all images from the source directory
        src_files = pcv.io.read_dataset(source_path=src)
        print(f"Processing {len(src_files)} images...")

    try:
        images = read_images(src_files)
        print(f"Images: {images}")
    except Exception as e:
        print("Error reading images.", e)


if __name__ == "__main__":
    main()
