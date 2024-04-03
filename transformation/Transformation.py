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
    blur: numpy.ndarray = None
    mask: numpy.ndarray = None
    roi: numpy.ndarray = None
    analyse: numpy.ndarray = None
    pseudolandmarks: numpy.ndarray = None
    color: numpy.ndarray = None


@dataclass
class Config:
    blur: bool
    mask: bool
    roi: bool
    analyse: bool
    pseudolandmarks: bool
    color: bool
    src: str
    dst: str


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
        default="data/processed/",
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


def write_images(dst: str, images: List[PcvImage]) -> None:
    print("Writing images...")
    for image in images:
        parts = image.img_name.split(".")
        # print(f"Writing image: {dst}/{image.img_name}")
        pcv.print_image(img=image.img, filename=f"{dst}/{image.img_name}")
        if image.blur is not None:
            new_file_path = (
                ".".join(parts[:-1]) + "_" + "blur" + "." + parts[-1]
            )
            pcv.print_image(img=image.blur, filename=f"{dst}/{new_file_path}")
        if image.mask is not None:
            new_file_path = (
                ".".join(parts[:-1]) + "_" + "mask" + "." + parts[-1]
            )
            pcv.print_image(img=image.mask, filename=f"{dst}/{new_file_path}")
        if image.roi is not None:
            new_file_path = (
                ".".join(parts[:-1]) + "_" + "roi" + "." + parts[-1]
            )
            pcv.print_image(img=image.roi, filename=f"{dst}/{new_file_path}")


def apply_transformation(image: PcvImage, config: Config) -> PcvImage:
    gray_img = pcv.rgb2gray_hsv(rgb_img=image.img, channel="s")
    binary_img = pcv.threshold.binary(
        gray_img=gray_img, threshold=36, object_type="dark"
    )
    if config.blur:
        image.blur = pcv.gaussian_blur(
            image=binary_img, ksize=(51, 51), sigma_x=0
        )
    if config.mask:
        image.mask = pcv.apply_mask(
            img=image.img, mask=binary_img, mask_color="white"
        )
    if config.roi:
        image.roi, roi_hierarchy = pcv.roi.rectangle(
            img=image.img, x=0, y=0, h=100, w=100
        )
    return image


def main():

    args = parse_arguments()

    single_image = False
    src_files = []
    if args.filename:
        print("Processing single image...", args.filename)
        single_image = True
        src_files.append(args.filename)
    if args.src:
        print("Source directory specified.", args.src)
        # Read all images from the source directory
        src_files = pcv.io.read_dataset(source_path=args.src)
        print(f"Processing {len(src_files)} images...")

    dst = args.dst.removesuffix("/")

    config = Config(
        blur=args.blur if not single_image else True,
        mask=args.mask if not single_image else True,
        roi=args.roi if not single_image else True,
        analyse=args.analyse if not single_image else True,
        pseudolandmarks=args.pseudolandmarks if not single_image else True,
        color=args.color if not single_image else True,
        src=src_files,
        dst=dst,
    )
    print("Configuration:")
    print("Source files:", config.src)
    print("Destination directory:", config.dst)
    print("Selected flags:")
    print("Mask processing:", config.mask)
    print("ROI objects processing:", config.roi)
    print("Analyse object processing:", config.analyse)
    print("Pseudolandmarks processing:", config.pseudolandmarks)
    print("Color histogram processing:", config.color)
    print("Applying image transformation...")

    try:
        images = read_images(src_files)
        # print(f"Images: {images}")
    except Exception as e:
        print("Error reading images.", e)
        return
    try:
        write_images(dst, images)
    except Exception as e:
        print("Error writing images.", e)
        return


if __name__ == "__main__":
    main()
