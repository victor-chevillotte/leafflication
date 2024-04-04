# import argparse
import numpy as np

# from plantcv import plantcv as pcv
# from plantcv.parallel import WorkflowInputs
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential


def main():
    try:
        dir = "./data/images"

        batch_size = 32
        img_height = 256
        img_width = 256
        seed = 123  # for reproducibility of the train dataset and the validation dataset

        train_data = tf.keras.utils.image_dataset_from_directory(
            dir,
            validation_split=0.2,
            subset="training",
            seed=seed,
            image_size=(img_height, img_width),
            batch_size=batch_size,
        )

        validation_data = tf.keras.utils.image_dataset_from_directory(
            dir,
            validation_split=0.2,
            subset="validation",
            seed=seed,
            image_size=(img_height, img_width),
            batch_size=batch_size,
        )

        class_names = train_data.class_names
        print(f"Class names : {class_names}")

        # plt.figure(figsize=(10, 10))
        # for images, labels in train_data.take(1):
        #     for i in range(9):
        #         ax = plt.subplot(3, 3, i + 1)
        #         plt.imshow(images[i].numpy().astype("uint8"))
        #         plt.title(class_names[labels[i]])
        #         plt.axis("off")
        #         plt.show()

        # for image_batch, labels_batch in train_data:
        #     print(image_batch.shape)
        #     print(labels_batch.shape)
        #     break
        # print : (32, 256, 256, 3), batch_size, img_height, img_width, RVB
        # (32,)

        AUTOTUNE = tf.data.AUTOTUNE
        train_data = (
            train_data.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        )
        validation_data = validation_data.cache().prefetch(
            buffer_size=AUTOTUNE
        )
        # for performance
        # https://www.tensorflow.org/tutorials/images/classification?hl=fr#configure_the_dataset_for_performance

        normalization_layer = layers.Rescaling(1.0 / 255)  # 0 - 255 to 0 - 1
        normalized_train_data = train_data.map(
            lambda x, y: (normalization_layer(x), y)
        )
        normalized_validation_data = validation_data.map(
            lambda x, y: (normalization_layer(x), y)
        )
        # image_batch, labels_batch = next(iter(normalized_ds))
        # first_image = image_batch[0]
        # print(np.min(first_image), np.max(first_image))

        nb_classes = len(class_names)
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
                layers.Dense(nb_classes, activation="softmax"),
            ]
        )
        # check dropout layers

        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        model.summary()
        epochs = 6
        history = model.fit(
            normalized_train_data,
            validation_data=normalized_validation_data,
            epochs=epochs,
        )

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
        model.save("first_saved_model.keras")
        exit(0)
    except Exception as e:
        print(f"An error has occured : {e}")


if __name__ == "__main__":
    main()
