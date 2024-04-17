import argparse
import tensorflow as tf
import os
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import shutil
from trainUtils.AugmentData import AugmentData
from trainUtils.TransformData import TransformData
from trainUtils.utils import Utils


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
    train_data = (
        train_data.cache()
        .shuffle(1000)
        .prefetch(buffer_size=AUTOTUNE)
    )
    validation_data = validation_data.cache().prefetch(buffer_size=AUTOTUNE)
    # for performance
    # https://www.tensorflow.org/tutorials/images/classification?hl=fr#configure_the_dataset_for_performance

    normalization_layer = layers.Rescaling(1.0 / 255)  # 0 - 255 to 0 - 1
    normalized_train_data = train_data.map(
        lambda x, y: (normalization_layer(x), y)
    )
    normalized_validation_data = validation_data.map(
        lambda x, y: (normalization_layer(x), y)
    )
    return normalized_train_data, normalized_validation_data, class_names


def main():
    try:
        parser = argparse.ArgumentParser(description="Training")
        parser.add_argument("-d", type=str, help="Directory path")
        parser.add_argument("-n", type=str, help="Model name")
        parser.add_argument("--e", type=int, help="Epochs")
        parser.add_argument("--b", type=int, help="Batch size")
        parser.add_argument("--s", type=int, help="Seed")
        parser.add_argument(
            "--v",
            type=int,
            help="Percents of validation data"
        )
        parser.add_argument(
            "--a",
            type=int,
            help="Augment data with a minimum of images per class"
        )
        parser.add_argument(
            "--t",
            action="store_true",
            help="Don't transform data"
        )
        args = parser.parse_args()
        (
            dir_path,
            model_name,
            epochs,
            batch_size,
            seed,
            validation_data_percents,
            img_per_class,
            transform_data_flag,
        ) = Utils.parse_args(args)
        img_height = 256
        img_width = 256

        # Data augmentation
        augmentation_options = ["flipped", "rotated"]
        dir_for_training = "trainingData"
        if os.path.exists(dir_for_training):
            print(f"The folder {dir_for_training} already exists")
            shutil.rmtree(dir_for_training)
            print(f"The folder {dir_for_training} has been deleted")
        shutil.copytree(dir_path, dir_for_training)
        print(f"The folder {dir_path} has been copied to {dir_for_training}")
        print()
        print("----- Augmenting data -----")
        AugmentData.augment_data(
            dir_for_training,
            augmentation_options,
            img_per_class
        )

        # Data transformation
        if transform_data_flag and len(augmentation_options) > 0:
            print("----- Transforming data -----")
            TransformData.transform_data(dir_for_training)

        Utils.display_histogram_terminal(dir_for_training)

        # Training
        (
            normalized_train_data,
            normalized_validation_data,
            class_names
        ) = get_data(
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
                # layers.Rescaling(1./255), for normalization,
                # but we have already done it
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
        model.fit(
            normalized_train_data,
            validation_data=normalized_validation_data,
            epochs=epochs,
        )

        # display_history(history, epochs)

        model.save(f"{model_name}.keras")
    except Exception as e:
        print(f"An error has occured : {e}")


if __name__ == "__main__":
    main()
