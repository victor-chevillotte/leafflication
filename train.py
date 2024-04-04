# import argparse
# from plantcv import plantcv as pcv
# from plantcv.parallel import WorkflowInputs
import argparse
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from augmentation import read_image_file, flip_image, rotate_image, save_augmented_image
from distribution import get_images_count
import shutil


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
        # for reproducibility of the train dataset and the validation dataset
        seed = args.s
    else:
        seed = np.random.randint(0, 1000)
    if args.v:
        validation_data_percents = args.v
        if validation_data_percents < 10:
            raise Exception("Percents of validation data must be greater than 10")
        elif validation_data_percents > 50:
            raise Exception("Percents of validation data must be less than 50")
    else:
        validation_data_percents = 20
    return (
        dir_path,
        model_name,
        epochs,
        batch_size,
        seed,
        validation_data_percents / 100,
    )


def get_data(
    dir_path,
    batch_size,
    seed,
    validation_data_percents,
    img_height=256,
    img_width=256,
):
    print(f"Validation data percents : {validation_data_percents}")
    train_data = tf.keras.utils.image_dataset_from_directory(
        dir_path,
        validation_split=validation_data_percents,
        subset="training",
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )
    validation_data = tf.keras.utils.image_dataset_from_directory(
        dir_path,
        validation_split=validation_data_percents,
        subset="validation",
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )
    class_names = train_data.class_names
    print("Class names : ", class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_data = train_data.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    validation_data = validation_data.cache().prefetch(buffer_size=AUTOTUNE)
    # for performance
    # https://www.tensorflow.org/tutorials/images/classification?hl=fr#configure_the_dataset_for_performance

    normalization_layer = layers.Rescaling(1.0 / 255)  # 0 - 255 to 0 - 1
    normalized_train_data = train_data.map(lambda x, y: (normalization_layer(x), y))
    normalized_validation_data = validation_data.map(
        lambda x, y: (normalization_layer(x), y)
    )
    return normalized_train_data, normalized_validation_data, class_names


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


def augmentation(image, file_path, augmentation_options):
    if "flipped" in augmentation_options:
        flipped_image = flip_image(image)
        save_augmented_image(flipped_image, file_path, "flipped")
        # print(f"Flipped image saved to {file_path}")
    if "rotated" in augmentation_options:
        rotated_image = rotate_image(image)
        save_augmented_image(rotated_image, file_path, "rotated")
        # print(f"Rotated image saved to {file_path}")
    # distorted_image = distort_image(image)
    # save_augmented_image(distorted_image, file_path, "distorted")


def augment_class(dir_path, count, img_per_class, augmentation_options):
    j = 0
    additional_images = img_per_class - count
    while j < additional_images:
        for i in range(len(augmentation_options)):
            for y in range(count + 1):
                image_path = f"{dir_path}/image ({y + 1}).JPG"
                image = read_image_file(image_path)
                augmentation(image, image_path, augmentation_options[i])
                j += 1
                if j >= additional_images:
                    return j


def augment_data(dir_path, img_per_class=600):
    augmentation_options = ["flipped", "rotated"]
    images_count = get_images_count(dir_path)
    counts = [dir["count"] for dir in images_count]
    if len(counts) <= 0:
        raise Exception("No images found")

    if img_per_class is None:
        img_per_class = max(counts)
    if img_per_class > min(counts) * len(augmentation_options):
        print(f"Each class will have {img_per_class} images")
        print(
            f"But minimum images per class is {min(counts)} images and we have {len(augmentation_options)} augmentation options"
        )
        img_per_class = min(counts) * len(augmentation_options)

    for img in images_count:
        if img["count"] < img_per_class:
            print(f"Augmenting {img['name']} class")
            nb_img_added = augment_class(
                img["path"], img["count"], img_per_class, augmentation_options
            )
            print(f"{nb_img_added} images added to {img['name']} class")
    print(f"Each class has now {img_per_class} images")
    return img_per_class


def main():
    try:
        parser = argparse.ArgumentParser(description="Prediction")
        parser.add_argument("-d", type=str, help="Directory path")
        parser.add_argument("-n", type=str, help="Model name")
        parser.add_argument("--e", type=int, help="Epochs")
        parser.add_argument("--b", type=int, help="Batch size")
        parser.add_argument("--s", type=int, help="Seed")
        parser.add_argument("--v", type=int, help="Percents of validation data")
        args = parser.parse_args()
        dir_path, model_name, epochs, batch_size, seed, validation_data_percents = (
            parse_args(args)
        )
        img_height = 256
        img_width = 256

        dir_for_training = "trainingData"
        if os.path.exists(dir_for_training):
            # raise Exception(f"The folder {dir_for_training} already exists")
            print(f"The folder {dir_for_training} already exists")
            shutil.rmtree(dir_for_training)
            print(f"The folder {dir_for_training} has been deleted")
        shutil.copytree(dir_path, dir_for_training)
        print(f"The folder {dir_path} has been copied to {dir_for_training}")

        min_img_per_class = augment_data(dir_for_training, img_per_class=600)
        normalized_train_data, normalized_validation_data, class_names = get_data(
            dir_for_training,
            batch_size,
            seed,
            validation_data_percents,
            img_height,
            img_width,
        )

        # CNN model
        model = Sequential(
            [
                layers.Input(shape=(img_height, img_width, 3)),
                # layers.Rescaling(1./255), for normalization, but we have already done it
                layers.Conv2D(16, 3, padding="same", activation="relu"),
                layers.MaxPooling2D(),
                layers.Conv2D(32, 3, padding="same", activation="relu"),
                layers.MaxPooling2D(),
                layers.Conv2D(64, 3, padding="same", activation="relu"),
                layers.MaxPooling2D(),
                layers.Flatten(),
                layers.Dense(128, activation="relu"),
                layers.Dense(len(class_names), activation="softmax"),
            ]
        )
        # check dropout layers

        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        model.summary()
        history = model.fit(
            normalized_train_data,
            validation_data=normalized_validation_data,
            epochs=epochs,
        )

        display_history(history, epochs)

        model.save(f"{model_name}.keras")
    except Exception as e:
        print(f"An error has occured : {e}")


if __name__ == "__main__":
    main()
