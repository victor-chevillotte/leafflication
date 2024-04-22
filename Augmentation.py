import argparse
import cv2
import numpy as np
import os


def read_image_file(file_path: str) -> np.ndarray:
    # Read the image from the given file path
    image = cv2.imread(file_path)
    if image is None:
        raise Exception("Image not found or unsupported image format")
    return image


def save_augmented_image(image, file_path, suffix):
    # Construct new file path with the suffix
    parts = file_path.split(".")
    new_file_path = ".".join(parts[:-1]) + "_" + suffix + "." + parts[-1]
    cv2.imwrite(new_file_path, image)


def flip_image(image):
    # Flip the image horizontally
    return cv2.flip(image, 1)


def rotate_image(image, angle=90):
    # Rotate the image by the given angle
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (width, height))


def shear_image(image):
    # Apply shear transformation
    rows, cols, _ = image.shape
    M = np.float32([[1, 0.1, 0], [0.1, 1, 0], [0, 0, 1]])
    return cv2.warpPerspective(image, M, (int(cols * 1.1), int(rows * 1.1)))


def crop_image(image):
    # Crop the central part of the image
    height, width = image.shape[:2]
    start_row, start_col = int(height * 0.07), int(width * 0.07)
    end_row, end_col = int(height * 0.90), int(width * 0.90)
    return image[start_row:end_row, start_col:end_col]


def distort_image(image):
    # Apply distortion to the image
    rows, cols = image.shape[:2]
    src_points = np.float32(
        [[0, 0], [cols - 1, 0], [0, rows - 1], [cols - 1, rows - 1]]
    )
    dst_points = np.float32(
        [
            [0, 0],
            [int(0.9 * cols), 0],
            [int(0.1 * cols), rows - 1],
            [cols - 1, rows - 1],
        ]
    )
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    return cv2.warpPerspective(image, M, (cols, rows))


def bright_image(image):
    # Apply contrast and brightness to the image
    return cv2.convertScaleAbs(image, alpha=1.7)


def blur_image(image):
    # Apply Gaussian blur to the image
    return cv2.GaussianBlur(image, (5, 5), 0)


def display_augmented_images(
    original_image,
    flipped_image,
    cropped_image,
    rotated_image,
    sheared_image,
    distorted_image,
    brighter_image,
    blurred_image,
):
    # Image names for display
    names = [
        "Original",
        "Flipped",
        "Cropped",
        "Rotated",
        "Sheared",
        "Distorted",
        "Brighter",
        "Blur",
    ]
    images = [
        original_image,
        flipped_image,
        cropped_image,
        rotated_image,
        sheared_image,
        distorted_image,
        brighter_image,
        blurred_image,
    ]

    # Standardizing images by adding text and padding
    standardized_images = []
    for img, name in zip(images, names):
        # Calculate the new width of the image to maintain aspect ratio
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
        current_end_x = current_x + img.shape[1]
        concatenated_image[: img.shape[0], current_x:current_end_x] = img
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


def augmentation(image, file_path, hide_display_option):
    # Apply each augmentation and save the augmented image
    flipped_image = flip_image(image)
    save_augmented_image(flipped_image, file_path, "flipped")
    cropped_image = crop_image(image)
    save_augmented_image(cropped_image, file_path, "cropped")
    rotated_image = rotate_image(image)
    save_augmented_image(rotated_image, file_path, "rotated")
    sheared_image = shear_image(image)
    save_augmented_image(sheared_image, file_path, "sheared")
    distorted_image = distort_image(image)
    save_augmented_image(distorted_image, file_path, "distorted")
    brighter_image = bright_image(image)
    save_augmented_image(brighter_image, file_path, "bright")
    blurred_image = blur_image(image)
    save_augmented_image(blurred_image, file_path, "blurred")

    if not hide_display_option:
        display_augmented_images(
            image,
            flipped_image,
            cropped_image,
            rotated_image,
            sheared_image,
            distorted_image,
            brighter_image,
            blurred_image,
        )


def augment_folder(
    folder_path: str, limit: int = None, hide_display_option: bool = False
):
    # Retrieve image files from the folder
    image_files = [
        f
        for f in os.listdir(folder_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    # Apply limit if specified
    if limit is not None:
        image_files = image_files[:limit]

    for file_name in image_files:
        file_path = os.path.join(folder_path, file_name)
        try:
            image = read_image_file(file_path)
            augmentation(image, file_path, hide_display_option)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Image augmentation script")
    parser.add_argument(
        "image_path",
        nargs="?",
        help="Path to the image file",
    )
    parser.add_argument(
        "-d",
        "--directory",
        help="Path to the directory containing images",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of images in directory to augment",
    )
    parser.add_argument(
        "--hide",
        action="store_true",
        help="Hide the display of augmented images",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.directory:
        augment_folder(args.directory, args.limit, args.hide)
    elif args.image_path:
        if os.path.isdir(args.image_path):
            augment_folder(args.image_path, args.limit, args.hide)
        else:
            image = read_image_file(args.image_path)
            augmentation(image, args.image_path, args.hide)
    else:
        print(
            "Usage: Augmentation.py [-d /path/directory"
            "[--limit number] [--hide]] | [image_path] [--hide]"
        )
        exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
