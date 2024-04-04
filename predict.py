import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model


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
        if not os.path.exists(dir):
            raise Exception("Directory doesn't exist")
    else:
        dir_path = None
    return model_path, image_path, dir_path


def predict_image(model, image_path, class_names):
    # Load the image
    img_height = 256
    img_width = 256
    image_pil = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(image_pil)
    img_array = tf.expand_dims(img_array, 0)

    # Predict the image
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    print(f"Predictions : {predictions}")
    print(f"Score : {score}")
    print(f"Predicted class : {class_names[np.argmax(score)]}")


def predict_directory(model, dir_path):
    seed = 123
    img_height = 256
    img_width = 256
    batch_size = 32
    validation_data = tf.keras.utils.image_dataset_from_directory(
        dir_path,
        validation_split=0.2,
        subset="validation",
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )
    normalization_layer = layers.Rescaling(1.0 / 255)  # 0 - 255 to 0 - 1
    normalized_validation_data = validation_data.map(
        lambda x, y: (normalization_layer(x), y)
    )

    class_names = validation_data.class_names
    score = model.evaluate(normalized_validation_data)
    print(f"Score : {score}")


def main():
    try:
        parser = argparse.ArgumentParser(description="Prediction")
        parser.add_argument("-m", type=str, help="Model file path")
        parser.add_argument("--i", type=str, help="Image file path")
        parser.add_argument("--d", type=str, help="Directory path")
        args = parser.parse_args()
        model_path, image_path, dir_path = parse_args(args)

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
            predict_directory(model, dir_path)
        else:
            raise Exception("No image or directory provided")

    except Exception as e:
        print(f"An error has occured : {e}")


if __name__ == "__main__":
    main()
