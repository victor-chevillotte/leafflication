import sys
import cv2
import numpy as np

def read_image_file(file_path: str) -> np.ndarray:
    # Read the image from the given file path
    image = cv2.imread(file_path)
    if image is None:
        raise Exception("Image not found or unsupported image format")
    return image

def save_augmented_image(image, file_path, suffix):
    # Construct new file path with the suffix
    parts = file_path.split('.')
    new_file_path = '.'.join(parts[:-1]) + '_' + suffix + '.' + parts[-1]
    cv2.imwrite(new_file_path, image)

def flip_image(image):
    # Flip the image horizontally
    return cv2.flip(image, 1)

def rotate_image(image, angle=90):
    # Rotate the image by the given angle
    height, width = image.shape[:2]
    center = (width//2, height//2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (width, height))

def shear_image(image):
    # Apply shear transformation
    rows, cols, _ = image.shape
    M = np.float32([[1, 0.2, 0], [0.2, 1, 0], [0, 0, 1]])
    return cv2.warpPerspective(image, M, (int(cols*1.2), int(rows*1.2)))

def crop_image(image):
    # Crop the central part of the image
    height, width = image.shape[:2]
    start_row, start_col = int(height*.25), int(width*.25)
    end_row, end_col = int(height*.75), int(width*.75)
    return image[start_row:end_row, start_col:end_col]

def distort_image(image):
    # Apply distortion to the image
    rows, cols = image.shape[:2]
    src_points = np.float32([[0,0], [cols-1,0], [0,rows-1], [cols-1,rows-1]])
    dst_points = np.float32([[0,0], [int(0.9*cols),0], [int(0.1*cols),rows-1], [cols-1,rows-1]])
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    return cv2.warpPerspective(image, M, (cols,rows))

def augmentation(image, file_path):
    # Apply each augmentation display resyult and save the augmented image
    augmented_image = flip_image(image)
    cv2.imshow("Flipped Image", augmented_image)
    save_augmented_image(augmented_image, file_path, "flipped")
    
    # Skew is a specific form of shearing, so this example uses shearing to represent both.

def main():
    try:
        if len(sys.argv) > 1:
            image = read_image_file(sys.argv[1])
            augmentation(image, sys.argv[1])
        else:
            print("Usage: python augmentation.py <path_to_image>")
            exit(1)

    except Exception as e:
        print(f"Error processing image: {e}")
        exit(1)

if __name__ == "__main__":
    main()
