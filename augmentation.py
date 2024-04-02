import sys
import cv2

def read_image_file(file_path : str) -> pd.DataFrame:
    try:
        df = cv2.imread(file_path)
        return df
    except Exception as e:
        print(f"Invalid Image file: {e}")
        exit(1)


def augmentation(df : pd.DataFrame) -> pd.DataFrame:
    try:
        # Augmentation code
        return df
    except Exception as e:
        print(f"Error in Augmentation: {e}")
        exit(1)


def main():
    try:
        if len(sys.argv) > 1:
            df = read_image_file(sys.argv[1])
        else:
            print("Usage: python augmentation.py <path_to_image>")
            exit(1)

    except Exception as e:
        print(f"Invalid CSV file: {e}")
        exit(1)


if __name__ == "__main__":
    main()
