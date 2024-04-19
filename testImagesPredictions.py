import subprocess
import os
import argparse
import re

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

RED = "\033[31m"
GREEN = "\033[32m"
BLUE = "\033[34m"
RESET = "\033[0m"


def testDir(dir_path, model_name):
    trueClasses = []
    predictedClasses = []
    files = os.listdir(dir_path)
    for file_name in files:
        file_path = os.path.join(dir_path, file_name)

        if os.path.isfile(file_path) and file_path.endswith(".JPG"):
            command = [
                "python3",
                "predict.py",
                "-m",
                model_name,
                "--i",
                file_path,
            ]

            result = subprocess.run(command, stdout=subprocess.PIPE, text=True)
            if result.stdout:
                lines = result.stdout.split("\n")
                for line in lines:
                    if "Predicted class :" in line:
                        predicted_class = line.split(":")[-1].strip()
                        filename_without_extension = re.sub(
                            r"\.jpe?g$", "", file_name, flags=re.IGNORECASE
                        )
                        cleaned_filename = re.sub(
                            r"\d+$", "", filename_without_extension
                        )
                        trueClasses.append(cleaned_filename)
                        predictedClasses.append(predicted_class)
                        break
            if result.stderr:
                print(f"Error for {file_name}:", result.stderr)
    return trueClasses, predictedClasses


def main():
    parser = argparse.ArgumentParser(description="Images test prediction")
    parser.add_argument(
        "-m", "--model", dest="m", type=str, help="Model name", required=True
    )
    parser.add_argument(
        "-d",
        "--directory",
        dest="d",
        required=True,
        type=str,
        help="Directory path",
    )
    args = parser.parse_args()
    model_name = f"{args.m}.keras" if not args.m.endswith(".keras") else args.m
    if args.d:
        folder_path = args.d
    else:
        folder_path = "data/test_images/Unit_test1"
    trueClasses, predictedClasses = testDir(folder_path, model_name)
    y = 0
    for i in range(len(trueClasses)):
        COLOR = RED
        if trueClasses[i] == predictedClasses[i]:
            y += 1
            COLOR = GREEN
        print(
            f"{COLOR}True class: {trueClasses[i]}, "
            f"Predicted class: {predictedClasses[i]}{RESET}"
        )
    COLOR = RED
    if len(trueClasses) == y:
        COLOR = GREEN
    print(f"{COLOR}Accuracy: {y}/{len(trueClasses)}{RESET}")


if __name__ == "__main__":
    main()
