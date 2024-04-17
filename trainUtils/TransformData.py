import os
import subprocess
from Transformation import (
    Config,
    write_images,
    apply_transformation,
    read_images
)
from distribution import get_images_count


class TransformData:

    @staticmethod
    def transform_data(dir_path):
        config = Config(
            blur=False,
            mask=True,
            roi=False,
            analyse=False,
            pseudolandmarks=False,
            color=False,
            src="",
            dst="",
        )

        images_class = get_images_count(dir_path)
        # images_class = [{path, name, count}]
        # example : images_class = [{'path': 'trainingData/Apple_Black_rot',
        # 'name': 'Apple_Black_rot', 'count': 620}]

        for img_class in images_class:  # for each class
            print(
                f"Transforming {img_class['name']} class, "
                f"{img_class['count']} images"
            )
            dir_path = img_class["path"]
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
            if os.path.isfile(file_path) and file_path.endswith(
                ".JPG"
            ):
                try:
                    image_path = file_path
                    image = read_images([image_path])
                    if image is None:
                        raise Exception(f"Image {image_path} not found")

                    # Apply transformation
                    try:
                        transformed_images = apply_transformation(
                            image[0],
                            config
                        )
                    except Exception:
                        error_transforming_count += 1

                    # Delete the original image
                    if os.path.exists(image_path):
                        try:
                            # subprocess.run(
                            #     ["rm", "-f", image_path],
                            #     check=True
                            # )
                            os.remove(image_path)
                        except subprocess.CalledProcessError as error:
                            print(f"Error deleting {image_path}: {error}")

                    # Save the transformed image
                    try:
                        if transformed_images:
                            write_images(
                                f"{dir_path}",
                                [transformed_images],
                            )
                        else:
                            print(f"Error transforming {image_path}")
                    except Exception as e:
                        print(f"Error writing {image_path}: {e}")

                except Exception as e:
                    print(f"Error transform_data {image_path}: {e}")
        if error_transforming_count > 0:
            print(
                f"Error transforming with"
                f"{error_transforming_count} images"
            )
