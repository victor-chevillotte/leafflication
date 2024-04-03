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
        predictions = model.predict(validation_data)
        print(f"Predictions : {predictions}")

        # Initialize lists to hold actual labels and predicted labels
        actual_labels = []
        predicted_labels = []
        for images, labels in validation_data:
            # each image is a batch of images (32 images in this case)
            for i in range(len(labels)):
                # get the actual label image by image
                actual_labels.append(labels[i])
                predicted_index = np.argmax(predictions[i])
                predicted_labels.append(predicted_index)
                predicted_class_name = validation_data.class_names[predicted_index]
                real_class_name = validation_data.class_names[labels[i]]
                print("=============================")
                print(f"Prediction : {predicted_class_name}")
                print(f"Real class : {real_class_name}")
      
        confusion_matrix = tf.math.confusion_matrix(actual_labels, predicted_labels)
        print(f"Confusion matrix:\n{confusion_matrix}")

        # Evaluate the model
        score = model.evaluate(validation_data)
        print(f"Score : {score}")
    except Exception as e:
        print(f"An error has occured : {e}")


if __name__ == "__main__":
    main()
