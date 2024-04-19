import os
import json
import numpy as np
import matplotlib.pyplot as plt
from Distribution import get_images_count
from dataclasses import dataclass


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
    patience: int = 2


class Utils:

    @staticmethod
    def parse_args(args):
        if args.d:
            dir_path = args.d
            if not os.path.exists(dir_path):
                raise Exception("Directory doesn't exist")
        else:
            raise Exception("Directory path is required")
        if args.n:
            model_name = args.n
        else:
            model_name = "first_model"
        if args.e:
            epochs = args.e
            if epochs < 1:
                raise Exception("Epochs must be greater than 0")
            elif epochs > 10:
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
