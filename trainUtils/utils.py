import os
import numpy as np
import matplotlib.pyplot as plt
from distribution import get_images_count


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
                raise Exception("Percents of validation data must be "
                                "greater than 10")
            elif validation_data_percents > 50:
                raise Exception("Percents of validation data must be "
                                "less than 50")
        else:
            validation_data_percents = 20
        if args.a:
            img_per_class = args.a
            if img_per_class < 1:
                raise Exception("Minimum images per class must be "
                                "greater than 0")
            elif img_per_class > 2000:
                raise Exception("Minimum images per class must be "
                                "less than 2000")
        else:
            img_per_class = 600
        if args.t:
            transform_data = False
        else:
            transform_data = True
        return (
            dir_path,
            model_name,
            epochs,
            batch_size,
            seed,
            validation_data_percents / 100,
            img_per_class,
            transform_data,
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
            values = [dir["count"] for dir in images_count]
            names = [dir["name"] for dir in images_count]
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
