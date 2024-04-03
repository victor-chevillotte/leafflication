# import argparse
import numpy as np
# from plantcv import plantcv as pcv
# from plantcv.parallel import WorkflowInputs
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model

def main():
    try:
        dir = "./data/images"
        model = load_model('first_saved_model.keras')
        seed = 123
        batch_size = 32
        img_height = 256
        img_width = 256
        validation_data = tf.keras.utils.image_dataset_from_directory(
            dir,
            validation_split = 0.2,
            subset = "validation",
            seed = seed,
            image_size = (img_height, img_width),
            batch_size = batch_size)

        normalization_layer = layers.Rescaling(1./255) # 0 - 255 to 0 - 1
        normalized_validation_data = validation_data.map(lambda x, y: (normalization_layer(x), y))

        class_names = validation_data.class_names
        print(f"Class names : {class_names}")
        predictions = model.predict(normalized_validation_data)
        print(f"Predictions : {predictions}")


        # Evaluate the model
        score = model.evaluate(normalized_validation_data)
        print(f"Score : {score}")

        
    except Exception as e:
        print(f"An error has occured : {e}")


if __name__ == "__main__":
    main()
