# DataScience X Logistic Regression

42 School Project


## Introduction

The Goal of this Computer Vision project is to classify 2 types of plant with different diseases using a Convolutional Neural Network (CNN) model


### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```bash
make
```

### Usage

#### Part I

```bash
python3 Distribution.py -d data/images/leaves
```

#### Part II

```bash
python3 Augmentation.py --folder data/images/leaves
```

#### Part III

- Visualize possible transformations of a single image
```bash
python3 Transformation.py <image_path>
```

- To apply specific transformations to all images in a folder
```bash
python3 Transformation.py -src data/images/leaves -dst data/images/processed -mask -analyse
```

#### Part IV

- Train the model
```bash
python3 Train.py -d data/leaves/images -n <model_name> --e <epoch_number> --b <batch_size>
```

- Predict the class of an image
```bash
python3 predict.py -i <image_path> -m <model_name>
```

## Data Exploration

### Data Analysis

The dataset contains the following plant categories :
- Apple (healthy)
- Apple (black rot)
- Apple (rust)
- Apple (scab)
- Grape (black rot)
- Grape (esca)
- Grape (healthy)
- Grape (spot)


### Data Visualization

#### Distribution

The script `Distribution.py` shows histograms for all the categories of the dataset

![](docs/distribution.png)

This allows us to target specific categories that are under represented in the dataset and apply data augmentation to balance the dataset

#### Augmentation

The script `Augmentation.py` shows the result of possible augmentations for the dataset

![](docs/augmentation.png)

We can specify a limit for the number of generated image in order to balance the dataset with the flag `--limit`

#### Transformation

The script `Transformation.py` shows the result of possible transformations for a single image when no `-src` flag is provided

![](docs/transformation.png)

- A binary mask is generated with the otsu thresholding method to isolate the leaf from the background
- A ROI (Region of Interest) is generated to crop the leaf from the image
- The leaf is analyzed with plantCV to extract its characteristics (area, perimeter, width, height, etc.)
- An histogram is provided to show the image color distribution of different color spaces (RGB, HSV and LAB)

When the `-src` flag is provided, the script applies the transformations to all images in the folder and saves the results in the destination folder
The available transformations are :
- Blur (gaussian blur applied to binary mask) `-blur`
- Masking (binary mask applied) `-mask`
- Pseudo Landmarks (shape analysis independant of the plant size) `-pseudolandmarks`
- ROI (Region of Interest applied) `-roi`
- Analysis (size and shape, applied on the ROI mask) `-analyse`
- Color Histogram (RGB, HSV and LAB histograms) `-color`

The `-dst` flag is used to specify the destination folder for the transformed images and all the transformations can be applied with the `-all` flag


## Classification

### Model Training




### Model Evaluation



## For correction 

```bash
python3 predict.py --i data/test_images/Unit_test1/Apple_Black_rot1.JPG  -m first_saved_model.keras
python3 predict.py --i data/test_images/Unit_test1/Apple_healthy1.JPG  -m first_saved_model.keras
python3 predict.py --i data/test_images/Unit_test1/Apple_healthy2.JPG  -m first_saved_model.keras
python3 predict.py --i data/test_images/Unit_test1/Apple_rust.JPG  -m first_saved_model.keras      
python3 predict.py --i data/test_images/Unit_test1/Apple_scab.JPG  -m first_saved_model.keras      

python3 predict.py --i data/test_images/Unit_test2/Grape_Black_rot1.JPG  -m first_saved_model.keras
python3 predict.py --i data/test_images/Unit_test2/Grape_Black_rot2.JPG  -m first_saved_model.keras
python3 predict.py --i data/test_images/Unit_test2/Grape_Esca.JPG  -m first_saved_model.keras   
python3 predict.py --i data/test_images/Unit_test2/Grape_healthy.JPG  -m first_saved_model.keras
python3 predict.py --i data/test_images/Unit_test2/Grape_spot.JPG  -m first_saved_model.keras
```
