import os
import json


def main():
    try:
        accuracies = []
        dir_path = "models_config_saved"

        for filename in os.listdir(dir_path):
            if filename.endswith(".json"):
                filepath = os.path.join(dir_path, filename)
                with open(filepath, 'r') as file:
                    data = json.load(file)
                    validation_accuracy = data['score']['validation_accuracy']
                    accuracies.append((filename, validation_accuracy))
        accuracies.sort(key=lambda x: x[1], reverse=True)
        for model in accuracies[:3]:
            print(f"Model : {model[0]} - Accuracy : {model[1]}")

    except Exception as e:
        print(f"An error has occured : {e}")


if __name__ == "__main__":
    main()
