import pandas as pd
import numpy as np
from logistic_regression import LogisticRegression


def extract_data(dataset: pd.DataFrame) -> (np.ndarray, np.ndarray):
    labels_data = np.array(dataset["Hogwarts House"])
    features = np.array(
        dataset[
            [
                "Arithmancy",
                # "Care of Magical Creatures",
                # "Defense Against the Dark Arts",
                "Muggle Studies",
                "Herbology",
                "Divination",
                "Ancient Runes",
                "Charms",
                "Astronomy",
                "History of Magic",
                "Transfiguration",
                "Potions",
                "Flying",
            ]
        ]
    )
    features = np.where(
        np.isnan(features), np.nanmean(features, axis=0), features
    )
    return (labels_data, features)


def pre_process_data(features: np.ndarray) -> np.ndarray:

    features = (features - np.mean(features, axis=0)) / np.std(
        features, axis=0
    )
    return features


def save_result(file: str, weights: list):
    np.save(file, np.array(weights, dtype="object"))


def read_csv(path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, index_col="Index")
    except FileNotFoundError:
        print("File not found")
        exit(1)
    except Exception as e:
        print(f"Invalid CSV file: {e}")
        exit(1)
    return df


def main():
    print("Training...")
    dataset: pd.DataFrame = read_csv("data/dataset_train.csv")
    labels, features = extract_data(dataset)
    features: np.ndarray = pre_process_data(features)
    trainer = LogisticRegression(learning_rate=0.01, max_iterations=1000)
    weights = trainer.fit(features, labels)
    save_result("data/weights", weights)

    print(
        "Score obtained on training data: "
        + str(trainer.score(features, labels) * 100)[:5]
        + "%"
    )

    trainer.plot_cost()


if __name__ == "__main__":
    main()
