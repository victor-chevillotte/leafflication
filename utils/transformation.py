import os
import subprocess
from Transformation import (
    Config,
    write_images,
    apply_transformation,
    read_images,
)
from utils.utils import get_images_count


class TransformData:

    @staticmethod
    def transform_data(dir_path, option="mask"):
        config = Config(
            blur=True if option == "blur" else False,
            mask=True if option == "mask" else False,
            roi=True if option == "roi" else False,
            analyse=True if option == "analyse" else False,
            pseudolandmarks=True if option == "pseudolandmarks" else False,
            color=True if option == "color" else False,
            src="",
            dst="",
        )

        images_class = get_images_count(dir_path)
        for category in images_class:
            print(
                f"Transforming {category.name} class, "
                f"{category.count} images"
            )
            dir_path = category.path
            TransformData.transform_class(dir_path, config)
        print()
        print()

    def transform_class(dir_path, config):
        error_transforming_count = 0
        # get all files in the directory
        for element in os.listdir(dir_path):
            # build the path of the file
            file_path = os.path.join(dir_path, element)
            # check if the file is a .JPG file
            if os.path.isfile(file_path) and file_path.endswith(".JPG"):
                try:
                    image_path = file_path
                    image = read_images([image_path])
                    if image is None:
                        raise Exception(f"Image {image_path} not found")

                    # Apply transformation
                    try:
                        transformed_images = apply_transformation(
                            image[0], config
                        )
                    except Exception:
                        error_transforming_count += 1

                    # Delete the original image
                    if os.path.exists(image_path):
                        try:
                            os.remove(image_path)
                        except subprocess.CalledProcessError as error:
                            print(f"Error deleting {image_path}: {error}")

                    # Save the transformed image
                    try:
                        if transformed_images:
                            write_images(
                                f"{dir_path}", [transformed_images], config
                            )
                        else:
                            print(f"Error transforming {image_path}")
                    except Exception as e:
                        print(f"Error writing {image_path}: {e}")

                except Exception as e:
                    print(f"Error transform_data {image_path}: {e}")
        if error_transforming_count > 0:
            print(
                f"Error transforming with" f"{error_transforming_count} images"
            )