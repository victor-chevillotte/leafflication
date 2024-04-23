import os
import json


def main():
    try:
        accuracies = []
        dir_path = "models_config_saved"

        for filename in os.listdir(dir_path):
            if filename.endswith(".json"):
                filepath = os.path.join(dir_path, filename)
                try:
                    with open(filepath, "r") as file:
                        data = json.load(file)
                        validation_accuracy = data["score"][
                            "validation_accuracy"
                        ]
                        if (
                            validation_accuracy is not None
                            and validation_accuracy > 0.9
                        ):
                            note = data.get("note")
                            if note is not None:
                                note = int(note.split("/")[0])
                            else:
                                note = -1
                            accuracies.append(
                                (filename, validation_accuracy, note)
                            )
                except Exception as e:
                    print(f"An error has occured : {e}, in {filepath}")
        accuracies.sort(key=lambda x: (x[2], x[1]), reverse=True)
        for model in accuracies:
            print(
                f"Model: {model[0]} - Accuracy: {model[1]}, Note: {model[2]}"
            )

    except Exception as e:
        print(f"An error has occured : {e}, in ")


if __name__ == "__main__":
    main()
