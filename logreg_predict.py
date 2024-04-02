import sys
import numpy as np
import pandas as pd
from logistic_regression import LogisticRegression
from logreg_train import (
    extract_data,
    read_csv,
    pre_process_data,
)


def save_filled_csv(file: str, df_test: pd.DataFrame, res: list):
    df_test["Hogwarts House"] = res
    df_test = pd.DataFrame(df_test, columns=["Hogwarts House"])
    try:
        df_test.to_csv("data/houses.csv")
    except Exception as e:
        print(f"Error writing houses.csv: {e}")


def get_weights(file: str) -> list:
    try:
        weights = np.load(file, allow_pickle=True)
    except Exception:
        print("Error: weights file not found")
        sys.exit(1)
    return weights


def main(argv):
    dataset = read_csv("data/dataset_test.csv")
    weights = get_weights("data/weights.npy")
    log_reg = LogisticRegression(learning_rate=1, max_iterations=1)
    log_reg.weights = weights
    labels, features = extract_data(dataset)
    prediction_vars = pre_process_data(features)
    houses = log_reg._predict(prediction_vars)
    save_filled_csv("data/houses.csv", dataset, houses)


if __name__ == "__main__":
    main(sys.argv)
