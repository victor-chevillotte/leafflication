import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import shutil
import random
import re
from dataclasses import dataclass
from matplotlib.backends.backend_agg import FigureCanvasBase
from typing import List

IMAGE_EXTENSIONS = [".jpg"]


@dataclass
class ImageCategory:
    path: str
    name: str
    count: int


def get_images_count(dir_path: str) -> List[ImageCategory]:
    images_count = []
    for path, _, files in os.walk(dir_path):
        dir_count = 0
        for file in files:
            for ext in IMAGE_EXTENSIONS:
                if file.endswith(ext.upper()):
                    dir_count += 1
                    break
        if dir_count > 0:
            images_count.append(
                ImageCategory(
                    path=path, name=path.split("/")[-1], count=dir_count
                )
            )
    return images_count


def plt_to_numpy_image(canvas: FigureCanvasBase) -> np.ndarray:
    canvas.draw()
    image_flat = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
    image = image_flat.reshape(*reversed(canvas.get_width_height()), 3)
    return image


def cv2_imshow_wrapper(window_name: str, image: np.ndarray):
    cv2.imshow(window_name, image)
    wait_time = 1000
    while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
        keyCode = cv2.waitKey(wait_time)
        if (keyCode & 0xFF) == ord("q"):
            cv2.destroyAllWindows()
            break


def pyqt_setup():
    VIRTUAL_ENV_PATH = os.environ.get("VIRTUAL_ENV")
    if VIRTUAL_ENV_PATH is None:
        print("Virtual environment not found.")
        print("Please activate the virtual environment.")
        exit(1)
    QT_PLUGIN_PATH = os.path.join(
        VIRTUAL_ENV_PATH,
        "lib/python3.10/site-packages/cv2/qt/plugins/platforms",
    )
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QT_PLUGIN_PATH


@dataclass
class ModelParameters:
    dir_path: str
    model_name: str
    epochs: int
    batch_size: int
    seed: int
    validation_data: int
    img_per_class: int
    transform_data_flag: bool
    augment_data_flag: bool
    augment_options: list = None
    transform_option: str = None
    img_size: tuple = (256, 256)
    patience: int = 10
    train_dataset: str = None
    validation_dataset: str = None


class Utils:

    @staticmethod
    def parse_args(args):
        dir_path = None
        if args.d:
            dir_path = args.d
            if not os.path.exists(dir_path):
                raise Exception("Directory doesn't exist")
            if args.validation_dataset or args.train_dataset:
                raise Exception(
                    "You can't use the -d and specified datasets options"
                    " at the same time"
                )
        elif not args.train_dataset and not args.validation_dataset:
            raise Exception("Directory path is required")
        if args.n:
            model_name = args.n
        else:
            model_name = "first_model"
        if args.e:
            epochs = args.e
            if epochs < 1:
                raise Exception("Epochs must be greater than 0")
            elif epochs > 50:
                raise Exception("Epochs must be less than 10")
        else:
            epochs = 6
        if args.b:
            batch_size = args.b
            if batch_size < 1:
                raise Exception("Batch size must be greater than 0")
            elif batch_size > 128:
                raise Exception("Batch size must be less than 128")
        else:
            batch_size = 32
        if args.s:
            # for reproducibility of the train dataset
            # and the validation dataset
            seed = args.s
        else:
            seed = np.random.randint(0, 1000)
        if args.v:
            validation_data_percents = args.v
            if validation_data_percents < 10:
                raise Exception(
                    "Percents of validation data must be " "greater than 10"
                )
            elif validation_data_percents > 50:
                raise Exception(
                    "Percents of validation data must be " "less than 50"
                )
        else:
            validation_data_percents = 20
        if args.a:
            img_per_class = args.a
            if img_per_class < 1:
                raise Exception(
                    "Minimum images per class must be " "greater than 0"
                )
            elif img_per_class > 2000:
                raise Exception(
                    "Minimum images per class must be " "less than 2000"
                )
        else:
            img_per_class = 600
        if args.t:
            transform_data = False
        else:
            transform_data = True
        if args.r:
            augment_data = False
        else:
            augment_data = True
        if not os.path.exists("models"):
            os.makedirs("models")
        if not os.path.exists("models_config_saved"):
            os.makedirs("models_config_saved")
        if args.train_dataset:
            train_dataset = args.train_dataset
            if not args.validation_dataset:
                raise Exception("Validation dataset is required")
        else:
            train_dataset = None
        if args.validation_dataset:
            validation_dataset = args.validation_dataset
            if not args.train_dataset:
                raise Exception("Train dataset is required")
        else:
            validation_dataset = None
        return ModelParameters(
            dir_path,
            model_name,
            epochs,
            batch_size,
            seed,
            validation_data_percents / 100,
            img_per_class,
            transform_data,
            augment_data,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
        )

    @staticmethod
    def display_history(history, epochs):
        acc = history.history["accuracy"]
        val_acc = history.history["val_accuracy"]
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        epochs_range = range(epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label="Training Accuracy")
        plt.plot(epochs_range, val_acc, label="Validation Accuracy")
        plt.legend(loc="lower right")
        plt.title("Training and Validation Accuracy")

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label="Training Loss")
        plt.plot(epochs_range, val_loss, label="Validation Loss")
        plt.legend(loc="upper right")
        plt.title("Training and Validation Loss")
        plt.show()

    @staticmethod
    def display_histogram_terminal(dir_path):
        try:
            images_count = get_images_count(dir_path)
            values = [category.count for category in images_count]
            names = [category.name for category in images_count]
            if len(values) != len(names):
                raise Exception("Values and names must have the same length")
            if len(values) <= 0 or len(names) <= 0:
                raise Exception("No data found")
            max_value = max(values)
            hist_height = 20
            char_for_bar = "="
            print("For training, we will use this data :")
            print()
            for i in range(len(values)):
                bar_length = int((values[i] / max_value) * hist_height)
                bar = char_for_bar * bar_length
                padding = " " * (hist_height - bar_length)
                print(f"{bar}{padding} : {values[i]} - {names[i]}")
            print()
            print()
        except Exception as e:
            print(f"Error display_histogram_terminal: {e}")

    @staticmethod
    def save_model(model, model_name, params, train_score, validation_score):
        try:
            if os.path.exists(f"models/{model_name}.keras"):
                model_name = model_name + "_1"
                while os.path.exists(f"models/{model_name}.keras"):
                    model_name = model_name[:-1] + str(int(model_name[-1]) + 1)
            model.save(f"models/{model_name}.keras")

            compile_info = {
                "optimizer": str(model.optimizer.__class__.__name__),
                "loss": model.loss,
                "metrics": model.metrics_names,
                "optimizer_config": model.optimizer.get_config(),
            }

            parameters = {
                "epochs": params.epochs,
                "batch_size": params.batch_size,
                "seed": params.seed,
                "validation_data_percents": params.validation_data,
                "img_per_class": params.img_per_class,
                "transform_data_flag": params.transform_data_flag,
                "augment_data_flag": params.augment_data_flag,
                "augment_options": params.augment_options,
                "transform_options": params.transform_option,
                "img_size": params.img_size,
                "patience": params.patience,
            }

            score = {
                "train_loss": train_score[0],
                "train_accuracy": train_score[1],
                "validation_loss": validation_score[0],
                "validation_accuracy": validation_score[1],
            }

            model_info = {
                "score": score,
                "parameters": parameters,
                "compile_infos": compile_info,
                "architecture": [],
            }

            for layer in model.layers:
                layer_config = layer.get_config()
                layer_info = {
                    "type": layer.__class__.__name__,
                    "units": layer_config.get("units"),
                    "activation": layer_config.get("activation"),
                    "padding": layer_config.get("padding"),
                }
                model_info["architecture"].append(layer_info)

            model_json = json.dumps(model_info, indent=4)
            with open(
                f"models_config_saved/{model_name}.json", "w"
            ) as json_file:
                json_file.write(model_json)

        except Exception as e:
            print(f"Error save_model: {e}")

    @staticmethod
    def split_data(
        dir_path,
        train_dir_path,
        validation_dir_path,
        validation_data,
        img_per_class,
        augmentation_options,
    ):
        try:
            print("Splitting data")
            images_class = get_images_count(dir_path)
            counts = [category.count for category in images_class]
            if len(counts) <= 0:
                raise Exception("No images found")

            min_per_class = min(counts) - (min(counts) * validation_data)
            print(f"min_per_class: {min_per_class}")
            if img_per_class is None:
                img_per_class = max(counts)
            if img_per_class > min_per_class * (len(augmentation_options) + 1):
                print(f"Each class will have {img_per_class} images minimum")
                print(
                    f"But minimum images per class is {min_per_class} images "
                    f"and we have {len(augmentation_options)} "
                    f"augmentation options"
                )
                print(
                    f"So we will have "
                    f"{min_per_class * (len(augmentation_options) + 1)} "
                    f"images minimum per class"
                )
                img_per_class = min_per_class * (len(augmentation_options) + 1)

            if os.path.exists(train_dir_path):
                shutil.rmtree(train_dir_path)
            os.makedirs(train_dir_path)
            if os.path.exists(validation_dir_path):
                shutil.rmtree(validation_dir_path)
            os.makedirs(validation_dir_path)

            for class_name in os.listdir(dir_path):
                class_dir_path = os.path.join(dir_path, class_name)
                if os.path.isdir(class_dir_path):
                    train_class_dir = os.path.join(train_dir_path, class_name)
                    validation_class_dir = os.path.join(
                        validation_dir_path, class_name
                    )
                    if os.path.exists(train_class_dir):
                        shutil.rmtree(train_class_dir)
                    os.makedirs(train_class_dir)
                    if os.path.exists(validation_class_dir):
                        shutil.rmtree(validation_class_dir)
                    os.makedirs(validation_class_dir)
                    images = []
                    for img in os.listdir(class_dir_path):
                        if img.lower().endswith((".jpg")):
                            images.append(img)
                    random.shuffle(images)
                    validation_data_size = int(min(counts) * validation_data)

                    for i, img in enumerate(images):
                        if i < validation_data_size:
                            shutil.copy(
                                os.path.join(class_dir_path, img),
                                os.path.join(validation_class_dir, img),
                            )
                        else:
                            shutil.copy(
                                os.path.join(class_dir_path, img),
                                os.path.join(train_class_dir, img),
                            )
            print("Data has been split")
            print("Train data:")
            for class_name in os.listdir(train_dir_path):
                class_count = 0
                for img in os.listdir(
                    os.path.join(train_dir_path, class_name)
                ):
                    class_count += 1
                print(f"Class: {class_name} - {class_count} images")
            print()
            print("Validation data:")
            for class_name in os.listdir(validation_dir_path):
                class_count = 0
                for img in os.listdir(
                    os.path.join(validation_dir_path, class_name)
                ):
                    class_count += 1
                print(f"Class: {class_name} - {class_count} images")
            print()
            dir_for_saved_images = "learnings"
            dir_for_saved_images_train = os.path.join(
                dir_for_saved_images, "train"
            )
            dir_for_saved_images_validation = os.path.join(
                dir_for_saved_images, "validation"
            )
            if os.path.exists(dir_for_saved_images):
                shutil.rmtree(dir_for_saved_images)
            os.makedirs(dir_for_saved_images)
            shutil.copytree(train_dir_path, dir_for_saved_images_train)
            shutil.copytree(
                validation_dir_path, dir_for_saved_images_validation
            )
            archive_name = "learnings"
            if os.path.exists(archive_name + ".zip"):
                os.remove(archive_name + ".zip")
            shutil.make_archive(archive_name, "zip", dir_for_saved_images)
            if os.path.exists(archive_name + ".zip"):
                print(f"Archive {archive_name}.zip created")
                shutil.rmtree(dir_for_saved_images)
            else:
                print(f"Error creating archive {archive_name}.zip")
            print()
            print()
            return int(img_per_class)
        except Exception as e:
            print(f"Error split_data: {e}")
            exit(1)

    @staticmethod
    def save_images(origin_dir_path, files_path, save_dir_path, data_dir_path):
        try:
            count = 0
            if os.path.exists(save_dir_path):
                shutil.rmtree(save_dir_path)
            os.makedirs(save_dir_path)
            pattern = re.compile(
                rf"{data_dir_path}"
                rf"/(?P<class>[^/]+)/image \((?P<number>\d+)\)(?:_mask)?\.JPG"
            )
            for file_path in files_path:
                match = pattern.match(file_path)
                if match:
                    count += 1
                    image_class = match.group("class")
                    image_number = match.group("number")
                    original_file_path = os.path.join(
                        origin_dir_path,
                        image_class,
                        f"image ({image_number}).JPG",
                    )
                    class_dir_path = os.path.join(save_dir_path, image_class)
                    if not os.path.exists(class_dir_path):
                        os.makedirs(class_dir_path)
                    save_file_path = os.path.join(
                        class_dir_path, f"image ({image_number}).JPG"
                    )
                    shutil.copy(original_file_path, save_file_path)
            print(f"Saved {count} images")
        except Exception as e:
            print(f"Error save_images: {e}")
            exit(1)
