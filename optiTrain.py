import tensorflow as tf
import os
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import shutil
from keras_tuner import RandomSearch
from trainUtils.AugmentData import AugmentData
from trainUtils.TransformData import TransformData
from trainUtils.Utils import Utils
from train import get_data


def build_model(hp, img_height, img_width, len_class_names):
    model = Sequential(
        [
            layers.Input(shape=(img_height, img_width, 3)),
            # layers.Rescaling(1./255), for normalization,
            # but we have already done it
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.BatchNormalization(),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.BatchNormalization(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            layers.Dense(len_class_names, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def main():
    try:
        img_height = 256
        img_width = 256
        img_per_class = 1500
        transform_data_flag = True
        batch_size = 64
        seed = 42
        validation_data_percents = 0.2

        dir_path = "data/images"
        dir_for_training = "trainingData"

        if False:
            # Data augmentation
            augmentation_options = ["flipped", "rotated", "bright", "cropped"]
            if os.path.exists(dir_for_training):
                print(f"The folder {dir_for_training} already exists")
                shutil.rmtree(dir_for_training)
                print(f"The folder {dir_for_training} has been deleted")
            shutil.copytree(dir_path, dir_for_training)
            print(
                f"The folder {dir_path} has been copied to {dir_for_training}"
            )
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

        stop_early = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3
        )

        tuner = RandomSearch(
            lambda hp: build_model(
                hp,
                img_height,
                img_width,
                len(class_names)
            ),
            objective='val_accuracy',
            max_trials=10,
            executions_per_trial=3,
            directory='searchTemp',
            project_name='hparam_tuning'
        )

        tuner.search(
            normalized_train_data,
            epochs=50,
            validation_data=normalized_validation_data,
            callbacks=[stop_early]
        )

        best_model = tuner.get_best_models(num_models=1)[0]

        best_model.summary()

    except Exception as e:
        print(f"An error has occured : {e}")


if __name__ == "__main__":
    main()
