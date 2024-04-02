import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from tools import courses, houses


def read_csv(path):
    try:
        df = pd.read_csv(path, index_col="Index")
    except FileNotFoundError:
        print("File not found. Please set dataset_train.csv in data folder")
        exit(1)
    return df


def display_scatter_plot(path):
    try:
        df = read_csv(path)
    except FileNotFoundError:
        print("File not found. Please set dataset_train.csv in data folder")
        exit(1)
    except Exception as e:
        print(f"Invalid CSV file: {e}")
        exit(1)
    fig, axs = plt.subplots(len(courses), len(courses), figsize=(25, 12))
    colors = cm.rainbow(np.linspace(0, 1, len(houses)))
    # What are the two features that are similar ?
    for i, course_x in enumerate(courses):
        for j, course_y in enumerate(courses):
            for k, house in enumerate(houses):
                house_df = df[df["Hogwarts House"] == house]
                axs[i][j].scatter(
                    house_df[course_x],
                    house_df[course_y],
                    color=colors[k],
                    alpha=0.5,
                    label=house,
                    s=0.5,  # set size of scatter points
                )
                axs[i][j].tick_params(
                    left=False,
                    right=False,
                    labelleft=False,
                    labelbottom=False,
                    bottom=False,
                )

                # Set column titles
                if i == 0:
                    # Alternate placement for readability
                    if not j % 2 == 0:
                        axs[i][j].set_title(course_y, fontsize=8)

                # Set line titles
                if j == 0:
                    # Alternate placement for readability
                    if i % 2 == 0:
                        axs[i][j].set_ylabel(course_x, rotation=90, fontsize=8)

    # Add color legend
    legend = fig.legend(
        houses, loc="lower center", ncol=len(houses), fontsize=8
    )
    for handle in legend.legend_handles:
        handle.set_sizes([50])  # set size of scatter points in legend

    try:
        plt.show()
    except KeyboardInterrupt:
        print("closing figure...")


def main():
    display_scatter_plot("data/dataset_train.csv")
    print("What are the two features that are similar ?")
    print(
        "Defense Against the Dark Arts and Astronomy are the two features that are similar"
    )


if __name__ == "__main__":
    main()
