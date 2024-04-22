import argparse
import tensorflow as tf
import os
import shutil
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from utils.augmentation import AugmentData
from utils.transformation import TransformData
from utils.utils import Utils


def get_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument(
        "-d", "--directory", type=str, dest="d", help="Directory path"
    )
    parser.add_argument("-n", "--name", dest="n", type=str, help="Model name")
    parser.add_argument("-e", "--epoch", dest="e", type=int, help="Epochs")
    parser.add_argument("-b", "--batch", dest="b", type=int, help="Batch size")
    parser.add_argument("-s", "--seed", dest="s", type=int, help="Seed")
    parser.add_argument(
        "-v",
        "--validation",
        dest="v",
        type=int,
        help="Percents of validation data",
    )
    parser.add_argument(
        "-a",
        "--augment",
        dest="a",
        type=int,
        help="Augment data with a minimum of images per class",
    )
    parser.add_argument(
        "--no-transform",
        action="store_true",
        dest="t",
        help="Don't transform data",
    )
    parser.add_argument(
        "--no-augment",
        dest="r",
        action="store_true",
        help="Don't augment data",
    )
    parser.add_argument(
        "-p",
        "--patience",
        dest="p",
        type=int,
        help="Patience for early stopping",
    )
    return parser.parse_args()


def data_augmentation(model_parameters, dir_for_training):
    if model_parameters.augment_data_flag:
        if os.path.exists(dir_for_training):
            print(f"The folder {dir_for_training} already exists")
            shutil.rmtree(dir_for_training)
            print(f"The folder {dir_for_training} has been deleted")
        shutil.copytree(model_parameters.dir_path, dir_for_training)
        print(
            f"The folder {model_parameters.dir_path} has been copied "
            f"to {dir_for_training}"
        )
        print()
        print("----- Augmenting data -----")
        model_parameters.img_per_class = AugmentData.augment_data(
            dir_for_training,
            model_parameters.augment_options,
            model_parameters.img_per_class,
        )

        # Data transformation
        if (
            model_parameters.transform_data_flag
            and len(model_parameters.augment_options) > 0
        ):
            print("----- Transforming data -----")
            TransformData.transform_data(
                dir_for_training, model_parameters.transform_option
            )

    Utils.display_histogram_terminal(dir_for_training)


def get_data(
    dir_path,
    batch_size,
    seed,
    validation_data,
    img_height=256,
    img_width=256,
):
    print(f"Validation data : {validation_data}")
    train_data = tf.keras.utils.image_dataset_from_directory(
        dir_path,
        validation_split=validation_data,
        subset="training",
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )
    validation_data = tf.keras.utils.image_dataset_from_directory(
        dir_path,
        validation_split=validation_data,
        subset="validation",
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )
    class_names = train_data.class_names
    print("Class names : ", class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_data = (
        train_data.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
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


def build_model(img_height, img_width, nb_classes):
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
            layers.Dense(nb_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    try:
        args = get_args()
        model_parameters = Utils.parse_args(args)
        model_parameters.augment_options = [
            "flipped",
            "distorted",
            "sheared",
            "bright",
            "rotated",
            "blurred",
            "cropped",
        ]
        model_parameters.transform_option = "mask"
        model_parameters.img_size = (256, 256)
        dir_for_training = "trainingData"

        data_augmentation(model_parameters, dir_for_training)

        # Training
        (normalized_train_data, normalized_validation_data, class_names) = (
            get_data(
                dir_for_training,
                model_parameters.batch_size,
                model_parameters.seed,
                model_parameters.validation_data,
                model_parameters.img_size[0],
                model_parameters.img_size[1],
            )
        )

        model = build_model(
            model_parameters.img_size[0],
            model_parameters.img_size[1],
            len(class_names),
        )
        model.summary()
        stop_early = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=model_parameters.patience
        )
        model.fit(
            normalized_train_data,
            validation_data=normalized_validation_data,
            epochs=model_parameters.epochs,
            callbacks=[stop_early],
        )

        train_score = model.evaluate(normalized_train_data)
        print(f"Training score : {train_score}")
        validation_score = model.evaluate(normalized_validation_data)
        print(f"Validation score : {validation_score}")

        # display_history(history, epochs)
        model_name = (
            f"{model_parameters.model_name}_E{model_parameters.epochs}"
            f"-B{model_parameters.batch_size}"
            f"-A{model_parameters.img_per_class}"
        )
        if model_parameters.transform_data_flag:
            model_name += "T"
        Utils.save_model(
            model, model_name, model_parameters, train_score, validation_score
        )
    except Exception as e:
        print(f"An error has occured : {e}")


if __name__ == "__main__":
    main()
