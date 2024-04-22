import argparse
import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from Transformation import (
    Config,
    write_images,
    apply_transformation,
    read_images,
)

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt


def parse_args(args):
    if args.m:
        model_path = args.m
        if not os.path.exists(model_path):
            raise Exception("Model file doesn't exist")
    else:
        raise Exception("No model file provided")
    if args.i:
        image_path = args.i
        if not os.path.exists(image_path):
            raise Exception("Image file doesn't exist")
    else:
        image_path = None
    if args.d:
        dir_path = args.d
        if not os.path.exists(dir_path):
            raise Exception("Directory doesn't exist")
    else:
        dir_path = None
    return model_path, image_path, dir_path


def predict_image(model, image_path, class_names):
    # Load the image
    img_height = 256
    img_width = 256

    image = read_images([image_path])
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
    register_dir = "temp"
    transformed_images = apply_transformation(image[0], config)
    if transformed_images:
        write_images("temp", [transformed_images], config)
    split_image_path = image_path.split("/")[-1].split(".")[0]
    new_file_path = f"{split_image_path}_mask.JPG"
    transformed_image_path = f"{register_dir}/{new_file_path}"
    print(f"Transformed image path : {transformed_image_path}")

    image_pil = tf.keras.preprocessing.image.load_img(
        transformed_image_path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(image_pil)
    img_array = tf.expand_dims(img_array, 0)

    # Predict the image
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_class = class_names[np.argmax(score)]
    print(f"Predictions : {predictions}")
    print(f"Score : {score}")
    print(f"Predicted class : {predicted_class}")
    # open widow showing original image side to side with transformed image
    # below them showing the predicted class

    app = QApplication([])

    window = QWidget()

    window.setWindowTitle("PyQt Image Display Example")
    window.setGeometry(100, 100, 800, 600)  # Position x, y, width, height

    # Layouts
    mainLayout = QVBoxLayout()
    imagesLayout = QHBoxLayout()

    # Images
    image1 = QLabel()
    pixmap1 = QPixmap(image_path)
    image1.setPixmap(pixmap1)
    image1.setAlignment(Qt.AlignCenter)

    image2 = QLabel()
    pixmap2 = QPixmap(transformed_image_path)
    image2.setPixmap(pixmap2)
    image2.setAlignment(Qt.AlignCenter)

    # Adding images to the images layout
    imagesLayout.addWidget(image1)
    imagesLayout.addWidget(image2)

    # Title
    titleLabel = QLabel("=== DL Classification ===")
    titleLabel.setAlignment(Qt.AlignCenter)

    # Predicted class
    if predicted_class in image_path:
        predictedClassLabel = QLabel(f"predicted class = <font color='green'>{predicted_class}</font>")
    else:
        predictedClassLabel = QLabel(f"predicted class = <font color='red'>{predicted_class}</font>")
    predictedClassLabel.setAlignment(Qt.AlignCenter)

    # Adding widgets to the main layout
    mainLayout.addLayout(imagesLayout)
    mainLayout.addWidget(titleLabel)
    mainLayout.addWidget(predictedClassLabel)

    # Set the main layout of the window
    window.setLayout(mainLayout)
    window.show()

    app.exec_()


def predict_directory(model, dir_path, class_names):
    print(f"Predicting images in directory : {dir_path}")
    img_height = 256
    img_width = 256
    batch_size = 32
    seed = 123
    data_to_predict = tf.keras.utils.image_dataset_from_directory(
        dir_path,
        validation_split=0.2,
        subset="validation",
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=False,
    )
    normalization_layer = layers.Rescaling(1.0 / 255)  # 0 - 255 to 0 - 1
    normalized_data_to_predict = data_to_predict.map(
        lambda x, y: (normalization_layer(x), y)
    )

    predictions = model.predict(normalized_data_to_predict)

    # The real classes are in the second element of the data_to_predict tuples
    real_classes = np.concatenate([y for x, y in data_to_predict], axis=0)
    correct_predictions_for_display = 0
    ok = 0
    wrong = 0
    # Iterate through predictions and real classes
    for i, prediction in enumerate(predictions):
        predicted_index = np.argmax(tf.nn.softmax(prediction))
        real_class_index = real_classes[i]
        if predicted_index != real_class_index:
            correct_predictions_for_display = 0
            wrong += 1
            print(
                f"\nERROR :Predicted class: {class_names[predicted_index]}, "
                f"Real class: {class_names[real_class_index]}"
            )
        else:
            ok += 1
            correct_predictions_for_display += 1
            # show on same line correct prediction number
            # print(
            #     f"Predicted class: {class_names[predicted_index]}, "
            #     f"Real class: {class_names[real_class_index]}, "
            #     f"Correct predictions: {correct_predictions_for_display}",
            #     end="\r",
            # )
    print(f"\nCorrect predictions : {ok}, Wrong predictions : {wrong}")
    print(f"Accuracy : {math.floor(ok/(ok+wrong)*100)}%")

    score = model.evaluate(normalized_data_to_predict)
    print(f"Score : {score}")


def main():
    try:
        parser = argparse.ArgumentParser(description="Prediction")
        parser.add_argument("-m", type=str, help="Model file path")
        parser.add_argument("-i", type=str, help="Image file path")
        parser.add_argument("-d", type=str, help="Directory path")
        args = parser.parse_args()
        model_path, image_path, dir_path = parse_args(args)
        print(model_path, image_path, dir_path)
        class_names = [
            "Apple_Black_rot",
            "Apple_healthy",
            "Apple_rust",
            "Apple_scab",
            "Grape_Black_rot",
            "Grape_Esca",
            "Grape_healthy",
            "Grape_spot",
        ]
        # Load the model
        model = load_model(model_path)

        if image_path:
            predict_image(model, image_path, class_names)
        elif dir_path:
            predict_directory(model, dir_path, class_names)
        else:
            raise Exception("No image or directory provided")

    except Exception as e:
        print(f"An error has occured : {e}")


if __name__ == "__main__":
    main()
