import argparse
import numpy as np
from plantcv import plantcv as pcv
from typing import List
from dataclasses import dataclass
import cv2
import matplotlib.pyplot as plt


@dataclass
class PcvImage:
    img: np.ndarray
    path: str
    img_name: str
    blur: np.ndarray = None
    mask: np.ndarray = None
    roi: np.ndarray = None
    analyse: np.ndarray = None
    pseudolandmarks: np.ndarray = None
    color: np.ndarray = None
    grey_scale: np.ndarray = None
    binary_mask: np.ndarray = None


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
        if image.analyse is not None:
            new_file_path = (
                ".".join(parts[:-1]) + "_" + "analyse" + "." + parts[-1]
            )
            pcv.print_image(
                img=image.analyse, filename=f"{dst}/{new_file_path}"
            )
        if image.pseudolandmarks is not None:
            new_file_path = (
                ".".join(parts[:-1])
                + "_"
                + "pseudolandmarks"
                + "."
                + parts[-1]
            )
            pcv.print_image(
                img=image.pseudolandmarks, filename=f"{dst}/{new_file_path}"
            )
    print("Images written.")


def define_roi(image: PcvImage) -> PcvImage:
    roi_image = image.img.copy()
    image_width = roi_image.shape[1]
    image_height = roi_image.shape[0]
    roi = pcv.roi.rectangle(
        img=image.img, x=0, y=0, h=image_height, w=image_width
    )
    kept_mask = pcv.roi.filter(
        mask=image.binary_mask, roi=roi, roi_type="partial"
    )
    colored_masks = pcv.visualize.colorize_masks(
        masks=[kept_mask], colors=["green"]
    )
    roi_image = pcv.visualize.overlay_two_imgs(
        img1=roi_image, img2=colored_masks, alpha=0.5
    )
    cv2.line(
        img=roi_image,
        pt1=(0, 0),
        pt2=(0, image_height),
        color=(255, 0, 0),
        thickness=10,
    )
    cv2.line(
        img=roi_image,
        pt1=(0, 0),
        pt2=(image_width, 0),
        color=(255, 0, 0),
        thickness=10,
    )
    cv2.line(
        img=roi_image,
        pt1=(0, image_height),
        pt2=(image_width, image_height),
        color=(255, 0, 0),
        thickness=10,
    )
    cv2.line(
        img=roi_image,
        pt1=(image_width, 0),
        pt2=(image_width, image_height),
        color=(255, 0, 0),
        thickness=10,
    )

    image.roi = roi_image
    return kept_mask


def apply_transformation(image: PcvImage, config: Config) -> PcvImage:
    image.grey_scale = pcv.rgb2gray_cmyk(rgb_img=image.img, channel="y")
    image.binary_mask = pcv.threshold.binary(
        gray_img=image.grey_scale, threshold=60, object_type="light"
    )
    image.binary_mask = pcv.fill_holes(bin_img=image.binary_mask)
    if config.blur:
        image.blur = pcv.gaussian_blur(
            img=image.binary_mask, ksize=(3, 3), sigma_x=0
        )
    if config.mask:
        image.mask = pcv.apply_mask(
            img=image.img, mask=image.binary_mask, mask_color="white"
        )
    if config.roi:
        roi_mask = define_roi(image)

    if config.color:
        histogram_with_colors(image.img)

    if config.pseudolandmarks:
        # The function returns coordinates of top, bottom, center-left, and center-right points
        points = pcv.homology.x_axis_pseudolandmarks(
            img=image.img, mask=image.binary_mask
        )
        # If needed, draw landmarks on the image for visualization
        landmark_image = np.copy(image.img)
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
        for index, group in enumerate(points):
            for point in group:
                # Ensure 'point' is a tuple or list of length 2, representing (x, y) coordinates
                point = (int(point[0][0]), int(point[0][1]))
                cv2.circle(
                    landmark_image,
                    point,
                    radius=5,
                    color=colors[index],
                    thickness=2,
                )
            image.pseudolandmarks = landmark_image

    if config.analyse:
        shape_image = pcv.analyze.size(
            img=image.img, labeled_mask=roi_mask, n_labels=1
        )
        image.analyse = shape_image

    return image


def display_results(image: PcvImage) -> None:
    # Image names for display
    names = [
        "Original",
        "blur",
        "mask",
        "pseudo landmarks",
        "roi",
        "analyse",
        "color histogram"
    ]
    images = [
        image.img,
        image.blur,
        image.mask,
        image.pseudolandmarks,
        image.roi,
        image.analyse,
        image.color
    ]

    # Standardizing images by adding text and padding
    standardized_images = []
    for img, name in zip(images, names):
        # Check if the image is grayscale (single channel). If so, convert to RGB.
        if len(img.shape) == 2 or img.shape[2] == 1:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # Calculate the new width of the image to maintain aspect ratio, leaving space for padding
        height, width = img.shape[:2]
        padded_img = cv2.copyMakeBorder(
            img, 50, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
        cv2.putText(
            padded_img,
            name,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        standardized_images.append(padded_img)

    # Concatenate images with padding between them
    total_width = sum(image.shape[1] for image in standardized_images) + (
        10 * (len(standardized_images) - 1)
    )
    max_height = max(image.shape[0] for image in standardized_images)
    concatenated_image = np.full(
        (max_height, total_width, 3), 255, dtype=np.uint8
    )

    # Place images with padding
    current_x = 0
    for img in standardized_images:
        concatenated_image[
            : img.shape[0], current_x : current_x + img.shape[1]
        ] = img
        current_x += (
            img.shape[1] + 10
        )  # Move to the next position with 10px padding

    # Display the concatenated image
    cv2.imshow("Augmented Images", concatenated_image)
    wait_time = 1000
    while cv2.getWindowProperty("Augmented Images", cv2.WND_PROP_VISIBLE) >= 1:
        keyCode = cv2.waitKey(wait_time)
        if (keyCode & 0xFF) == ord("q"):
            cv2.destroyAllWindows()
            break


def histogram_with_colors(pcv_image: PcvImage):

    image = pcv_image.img
    histograms = []

    # Convert the image to different color spaces
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    lab = "lightness, green-magenta, blue-yellow"

    # Define the color spaces you want to analyze
    color_spaces = [
        ("blue", cv2.COLOR_BGR2RGB, rgb, 2),
        ("blue-yellow", cv2.COLOR_BGR2LAB, lab, 2),
        ("green", cv2.COLOR_BGR2RGB, rgb, 1),
        ("green-magenta", cv2.COLOR_BGR2LAB, lab, 1),
        ("hue", cv2.COLOR_BGR2HSV, hsv, 0),
        ("lightness", cv2.COLOR_BGR2LAB, lab, 0),
        ("red", cv2.COLOR_BGR2RGB, rgb, 0),
        ("saturation", cv2.COLOR_BGR2HSV, hsv, 1),
        ("value", cv2.COLOR_BGR2HSV, hsv, 2),
    ]

    # Plot the pixel intensity distribution
    fig = plt.figure(figsize=(10, 6))
    canvas = fig.canvas

    for color_space, conversion, channel, channel_index in color_spaces:
        # Convert image to desired color space
        converted_image = cv2.cvtColor(image, conversion)

        # Extract the specified channel
        extracted_channel = converted_image[:, :, channel_index]

        # Calculate histogram
        hist = cv2.calcHist([extracted_channel], [0], None, [256], [0, 256])

        # Normalize histogram
        hist /= hist.sum()
        hist *= 100

        histograms.append((color_space, hist))
        # Plot histogram
        plt.plot(hist, label=color_space)

    plt.title("Pixel Intensity Distribution for Different Color Spaces")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Proportion of Pixels (%)")
    plt.legend()
    plt.grid(True)
    canvas.draw()
    image_flat = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')  # (H * W * 3,)
    # NOTE: reversed converts (W, H) from get_width_height to (H, W)
    image = image_flat.reshape(*reversed(canvas.get_width_height()), 3)  # (H, W, 3)
    pcv_image.color = image
    # print(f"image: {image}")
    # cv2.imshow("Histograms", image)
    # plt.show()
    return histograms


def display_histogram(histograms):
    plt.figure(figsize=(20, 10))
    for color_space, hist in histograms:
        plt.plot(hist, label=color_space)
    plt.legend()
    plt.show()


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
    for image in images:
        apply_transformation(image, config)
    try:
        if single_image and len(images) == 1:
            print("Displaying results...")
            display_results(images[0])
            histo = histogram_with_colors(
                img=images[0].img,
                color_spaces=[
                    "blue",
                    "blue-yellow",
                    "green",
                    "green-magenta",
                    "hue",
                    "lightness",
                    "red",
                    "saturation",
                    "value",
                ],
            )
            display_histogram(histo)

        else:
            write_images(dst, images)
    except Exception as e:
        print("Error writing images.", e)
        return


if __name__ == "__main__":
    main()
