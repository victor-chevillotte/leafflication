import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tools import courses


def read_csv(path):
    try:
        df = pd.read_csv(path, index_col="Index")
    except FileNotFoundError:
        print("File not found. Please set dataset_train.csv in data folder")
        exit(1)
    except Exception as e:
        print(f"Invalid CSV file: {e}")
        exit(1)
    return df


def main():
    print("Preparing pairplot... (This may take a while)")
    df = read_csv("data/dataset_train.csv")
    df = df[[*courses, "Hogwarts House"]]
    sns.pairplot(df, hue="Hogwarts House")
    try:
        plt.show()
    except KeyboardInterrupt:
        print("closing figure...")
    print(
        "From this visualization, what features are you going to use for your logistic regression?"
    )


if __name__ == "__main__":
    main()
