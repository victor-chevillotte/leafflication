from augmentation import (
    read_image_file,
    flip_image,
    rotate_image,
    save_augmented_image
)
from distribution import get_images_count


class AugmentData:

    @staticmethod
    def augment_data(dir_path, augmentation_options, img_per_class=600):
        images_class = get_images_count(dir_path)
        # images_class = [{path, name, count}]
        # example : images_class = [{'path': 'trainingData/Apple_Black_rot',
        # 'name': 'Apple_Black_rot', 'count': 620}]
        counts = [dir["count"] for dir in images_class]
        if len(counts) <= 0:
            raise Exception("No images found")

        # Calculate the minimum number of images per class
        if img_per_class is None:
            img_per_class = max(counts)
        if img_per_class > min(counts) * len(augmentation_options):
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

        for img_class in images_class:
            # Augment the class if it has less than the minimum number of
            # images per class
            if img_class["count"] < img_per_class:
                print(
                    f"Augmenting {img_class['name']} class, "
                    f"{img_class['count']} images"
                )
                nb_img_added = AugmentData.augment_class(
                    img_class["path"],
                    img_class["count"],
                    img_per_class,
                    augmentation_options,
                )
                print(f"{nb_img_added} images added to {img_class['name']}")
        print(f"Each class has now {img_per_class} images minimum")
        print()
        return img_per_class

    @staticmethod
    def augment_class(dir_path, count, img_per_class, augmentation_options):
        j = 0
        additional_images = img_per_class - count
        while j < additional_images:
            for i in range(len(augmentation_options)):
                for y in range(count):
                    image_path = f"{dir_path}/image ({y + 1}).JPG"
                    image = read_image_file(image_path)
                    AugmentData.augmentation(
                        image,
                        image_path,
                        augmentation_options[i]
                    )
                    j += 1
                    if j >= additional_images:
                        return j

    @staticmethod
    def augmentation(image, file_path, augmentation_options):
        if "flipped" in augmentation_options:
            flipped_image = flip_image(image)
            save_augmented_image(flipped_image, file_path, "flipped")
        if "rotated" in augmentation_options:
            rotated_image = rotate_image(image)
            save_augmented_image(rotated_image, file_path, "rotated")
        # distorted_image = distort_image(image)
        # save_augmented_image(distorted_image, file_path, "distorted")
