import os
import re
from Augmentation import (
    read_image_file,
    flip_image,
    rotate_image,
    bright_image,
    crop_image,
    save_augmented_image,
)
from utils.utils import get_images_count


class AugmentData:

    @staticmethod
    def augment_data(
        dir_path: str, augmentation_options: list, img_per_class=600
    ):
        images_class = get_images_count(dir_path)
        counts = [category.count for category in images_class]
        if len(counts) <= 0:
            raise Exception("No images found")

        # Calculate the minimum number of images per class
        if img_per_class is None:
            img_per_class = max(counts)
        if img_per_class > min(counts) * (len(augmentation_options) + 1):
            print(f"Each class will have {img_per_class} images minimum")
            print(
                f"But minimum images per class is {min(counts)} images "
                f"and we have {len(augmentation_options)} augmentation options"
            )
            print(
                f"So we will have "
                f"{min(counts) * (len(augmentation_options) + 1)} "
                f"images minimum per class"
            )
            img_per_class = min(counts) * (len(augmentation_options) + 1)

        for category in images_class:
            # Augment the class if it has less than the minimum number of
            # images per class
            if category.count < img_per_class:
                print(
                    f"Augmenting {category.name} class, "
                    f"{category.count} images"
                )
                nb_img_added = AugmentData.augment_class(
                    category.path,
                    category.count,
                    img_per_class,
                    augmentation_options,
                )
                print(f"{nb_img_added} images added to {category.name}")
            else:
                AugmentData.remove_files(
                    category.path,
                    category.count - img_per_class
                )
        print(f"Each class has now {img_per_class} images minimum")
        print()
        return img_per_class

    @staticmethod
    def augment_class(dir_path, count, img_per_class, augmentation_options):
        j = 0
        additional_images = img_per_class - count
        if additional_images > 0:
            for i in range(len(augmentation_options)):
                for y in range(count):
                    image_path = f"{dir_path}/image ({y + 1}).JPG"
                    if os.path.exists(image_path):
                        try:
                            image = read_image_file(image_path)
                            AugmentData.augmentation(
                                image, image_path, augmentation_options[i]
                            )
                            j += 1
                            if j >= additional_images:
                                return j
                        except Exception:
                            None

    @staticmethod
    def augmentation(image, file_path, augmentation_options):
        if "flipped" in augmentation_options:
            flipped_image = flip_image(image)
            save_augmented_image(flipped_image, file_path, "flipped")
        if "rotated" in augmentation_options:
            rotated_image = rotate_image(image)
            save_augmented_image(rotated_image, file_path, "rotated")
        if "bright" in augmentation_options:
            brighter_image = bright_image(image)
            save_augmented_image(brighter_image, file_path, "bright")
        if "cropped" in augmentation_options:
            cropped_image = crop_image(image)
            save_augmented_image(cropped_image, file_path, "cropped")

    @staticmethod
    def remove_files(dir_path, nb_files_to_remove):
        try:
            print(f"Removing {nb_files_to_remove} files from {dir_path}")
            highest, jpg_count = AugmentData.get_count_and_highest_number(
                dir_path
            )
            for i in range(highest):
                file_path = f"{dir_path}/image ({highest - i}).JPG"
                if os.path.exists(file_path):
                    os.remove(file_path)
                    nb_files_to_remove -= 1
                if nb_files_to_remove <= 0:
                    break
        except Exception as e:
            print(f"Error while removing files: {e}")

    @staticmethod
    def get_count_and_highest_number(dir_path):
        highest = None
        jpg_count = 0
        pattern = r"image \((\d+)\)\.JPG"
        for file in os.listdir(dir_path):
            match = re.match(pattern, file)
            if match:
                number = int(match.group(1))
                jpg_count += 1
                if highest is None or number > highest:
                    highest = number
        return highest, jpg_count
