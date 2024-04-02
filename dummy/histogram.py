import pandas as pd
import matplotlib.pyplot as plt
from tools import courses, houses


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


def display_histogram(path):
    df = read_csv(path)
    fig, axs = plt.subplots(3, 5, figsize=(15, 8))
    for i, course in enumerate(courses):
        # Turn grid into array of length [number of features]
        axs = axs.flatten()[: len(courses)]
        for house in houses:
            axs[i].hist(
                df[df["Hogwarts House"] == house][course],
                alpha=0.5,
                label=house,
            )
        axs[i].set_title(course)

        # Set house legend in bottom right corner
    handles, labels = axs[0].get_legend_handles_labels()
    plt.legend(handles, labels, loc="center")
    fig.suptitle(
        "Distribution of scores for each course between all four houses"
    )
    fig.tight_layout()
    fig.subplots_adjust(
        top=0.93, hspace=0.2, wspace=0.15
    )  # margin between subplots
    try:
        plt.show()
    except KeyboardInterrupt:
        print("closing figure...")


def main():
    display_histogram("data/dataset_train.csv")
    print(
        "Which Hogwarts course has a homogeneous score distribution between all four houses?"
    )
    print(
        "Care of magical creatures has a homogeneous score distribution between all four houses"
    )


if __name__ == "__main__":
    main()
