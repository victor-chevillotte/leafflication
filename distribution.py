import argparse
import os
import matplotlib.pyplot as plt


IMAGE_EXTENSIONS = [".jpg"]


def create_bar_chart(values, names, title, xlabel, ylabel, colors, axis):
    bar_width = 0.8
    if len(values) == 1:
        bar_width = 0.1
    axis.bar(names, values, color=colors, width=bar_width)
    if len(values) == 1:
        axis.set_xlim([-0.5, 0.5])
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)


def create_pie_chart(values, names, title, colors, axis):
    axis.pie(values, labels=names, colors=colors, autopct="%1.1f%%")
    axis.set_title(title)


def create_text(ax, text):
    ax.text(0.5, 0.5, text, ha='center', va='center')
    ax.axis('off')

def get_images_count(dir_path):
    images_count = []
    for path, _, files in os.walk(dir_path):
        dir_count = 0
        for file in files:
            for ext in IMAGE_EXTENSIONS:
                if file.endswith(ext.upper()):
                    dir_count += 1
                    break
        if dir_count > 0:
            images_count.append(
                {"path": path, "name": path.split("/")[-1], "count": dir_count}
            )
    return images_count


def main():
    try:
        parser = argparse.ArgumentParser(description="Distribution")
        parser.add_argument("-d", type=str, help="Directory name")
        args = parser.parse_args()
        if args.d:
            dir_path = args.d
            if not os.path.exists(dir_path):
                raise Exception("Directory doesn't exist")
        else:
            raise Exception("No directory provided")
        images_count = get_images_count(dir_path)
        counts = [dir["count"] for dir in images_count]
        if len(counts) <= 0:
            raise Exception("No images found")
        names = [dir["name"] for dir in images_count]
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i % 20) for i in range(len(counts))]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 10))
        create_bar_chart(counts, names, "", "Directories", "Images count", colors, ax1)
        create_pie_chart(counts, names, "", colors, ax2)
        plt.show()
    except argparse.ArgumentError as e:
        print(f"Error with arguments: {e}")
    except Exception as e:
        print(f"An error has occured : {e}")


if __name__ == "__main__":
    main()
