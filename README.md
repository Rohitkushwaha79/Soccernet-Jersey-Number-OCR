# Soccernet-Jersey-Number-OCR
## Overview
This repository presents experiments conducted for Optical Character Recognition (OCR) of jersey numbers in soccer images. The experiments involve training and evaluating various models using datasets from both SVHN and SoccerNet Jersey Number Recognition. While models were trained on both datasets, the evaluation focused on the SoccerNet Jersey Number Recognition dataset. The performance of these models was assessed on the SoccerNet Jersey Number Recognition test and challenge datasets, with results reported accordingly.
## Table of Contents
  - [Structure](#structure)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Data Download](#data-download)
    - [Create Custom Soccernet Jersey Dataset](#2-create-custom-soccernet-jersey-dataset)
    - [Preprocessing](#2-preprocessing)
    - [Model Training](#3-model-training)
    - [Evaluation](#4-evaluation)
    - [Inference](#5-inference)
  - [Experiments](#experiments)
    - [Experiment 1: CTC CRNN Model on SVHN Dataset](#experiment-1-ctc-crnn-model-on-svhn-dataset)
    - [Experiment 2: CNN Model on Soccernet Dataset](#experiment-2-cnn-model-on-soccernet-dataset)
    - [Experiment 3: OCR with EasyOCR on Soccernet Challenge Dataset](#experiment-3-ocr-with-easyocr-on-soccernet-challenge-dataset)
    - [Experiment 4: CTC CRNN Model on Merged Dataset](#experiment-4-ctc-crnn-model-on-merged-dataset)
  - [Experiment Results](#experiment-results)
## Structure
- **Data:** Contains datasets used for training and evaluation, including SVHN and Soccernet datasets.
- **Models:** Holds pre-trained models and checkpoints.
- **Notebooks:** Jupyter notebooks for experimenting and exploring the data.
- **Src:** Source code directory containing the following subdirectories:
  - **Data:**
    - [svhn_data_loader.py](src/data/svhn_data_loader.py) :Loads SVHN dataset
    - [soccernet_data_loader.py](src/data/soccernet_data_loader.py) :Loads Custuon Soccernet Jerney number dataset
  - **Layer:**
    - [ctc_layer.py](src/layer/ctc_layer.py) :CTC loss layer implementation
  - **Models:**
    - [model.py](src/models/model.py): Contains code for the CTC-CRNN model.
  - **Script:**
    - [evaluate.py](src/script/evaluate.py): Script for evaluating the trained model on the Soccernet test dataset.
    - [test.py](src/script/test.py): Script for testing the model on images.
    - [train.py](src/script/train.py): Script for training the model on the merged training dataset.
    - [inference_on_soccernet_challenge_dataset_by_trained_model.py](src/script/inference_on_soccernet_challenge_dataset_by_trained_model.py): Script for running inference on the Soccernet Challenge dataset using the trained CTC-CRNN model.
    - [inference_on_soccernet_challenge_dataset_by_EasyOCR.py](src/script/inference_on_soccernet_challenge_dataset_by_EasyOCR.py): Script for running inference on the Soccernet Challenge dataset using EasyOCR.
  - **Utils:**
    - [custom_ctc_accuracy.py](src/utils/custom_ctc_accuracy.py)
    - [predict_majority_jersey_no_for_playerid.py](src/utils/predict_majority_jersey_no_for_playerid.py)
    - [merge_datasets.py](src/utils/merge_datasets.py)
    - [extract_jersey.py](src/utils/extract_jersey.py)

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Rohitkushwaha79/Soccernet-Jersey-Number-OCR.git

2. **Install Dependencies**
   * Create a virtual environment to manage project dependencies effectively (recommended).
   * Install the required libraries using `pip install -r requirements.txt`.


## Usage
### Data Download
- Download the SVHN dataset from [link](https://www.kaggle.com/datasets/stanfordu/street-view-house-numbers).
- Download the Soccernet dataset from [link](https://www.soccer-net.org/data#h.b1lf96jmxlcc).
- 
### 2. Create Custom Soccernet Jersey Dataset
- Generate the custom Soccernet jersey dataset using the notebook [SoccerNet_Jersey_Extraction_Dataset_Creation.ipynb](notebooks/SoccerNet_Jersey_Extraction_Dataset_Creation.ipynb).

### 2. Preprocessing
- Preprocess the datasets:
  - Run [svhn_data_loader.py](src/data/svhn_data_loader.py) to load and preprocess the SVHN dataset.
  - Run [soccernet_data_loader.py](src/data/soccernet_data_loader.py) to load and preprocess the Custom Soccernet dataset(contains jersey extracts with their corresponding label as their folder name). 

### 3. Model Training
- Train the model on the merged train dataset (SVHN and Custom Soccernet training datasets for jersey number recognition):
```bash
python src/script/train.py
```
### 4. Evaluation
- Evaluate the trained model on the Soccernet jersey number recognition test dataset:
```bash
python src/script/evaluate.py
```

### 5. Inference
- Run inference on the Soccernet Challenge dataset using the trained CTC-CRNN model:
```bash
python src/script/inference_on_soccernet_challenge_dataset_by_trained_model.py
```
- Alternatively, run inference using EasyOCR:
```bash
python src/script/inference_on_soccernet_challenge_dataset_by_EasyOCR.py
```
## Experiments

### Experiment 1: CTC CRNN Model on SVHN Dataset
- **Description:** Trained a CTC (Connectionist Temporal Classification) CRNN (Convolutional Recurrent Neural Network) model on the Street View House Numbers (SVHN) dataset.
- **Train Dataset Used:** SVHN dataset
- **Test Dataset Used:** Custom Soccernet test dataset
- **Accuracy on Test Dataset:** 39%

### Experiment 2: CNN Model on Custom Soccernet Dataset
- **Description:** Trained a Convolutional Neural Network (CNN) model on the Soccernet training dataset.
- **Train Dataset Used:** Custom Soccernet training dataset
- **Test Dataset Used:** Custom Soccernet test dataset
- **Accuracy on Test Dataset:** 24%

### Experiment 3: OCR with EasyOCR on Soccernet Challenge Dataset
- **Description:** Performed inference on the Soccernet Challenge dataset using EasyOCR.
  - **Steps involved:**
    1. Extracting jersey images using human pose estimation model (YOLOv8) and logic from the Soccernet Challenge dataset.
    2. Applying EasyOCR on the extracted jersey crops.
  - **Accuracy on Challenge Dataset:** 58%

### Experiment 4: CTC CRNN Model on Merged Dataset
- **Description:** Trained a CTC CRNN model on a merged dataset (SVHN and Soccernet training datasets).
  - **Steps involved:**
    1. Extracting jersey images using human pose estimation model (YOLOv8) and logic from Soccernet train and test datasets.
    2. Manual cleaning of the extracted datasets.
    3. Data preprocessing.
    4. Merging SVHN and Custom Soccernet data.
  - **Accuracy on Custom Training Dataset:** 98%
  - **Accuracy on Custom Test Dataset:** 93%
  - **Accuracy on Challenge Dataset:** 71%

## Experiment Results

| Model        | Train Dataset Used           | Test Dataset Used | Accuracy on Custom Soccernet Train Dataset | Accuracy on Custom Soccernet Test Dataset | Accuracy on Soccernet Jersey Number Recognition Challenge Dataset |
|--------------|-------------------------------|-------------------|---------------------------|--------------------------|-------------------------------|
| CTC CRNN     | SVHN                          | Soccernet         | 39%                       | -                        | -                             |
| CNN          | Soccernet                     | Soccernet         | -                         | 24%                      | -                             |
| EasyOCR      | -                             | Soccernet Challenge | -                       | -                        | 58%                           |
| CTC CRNN     | SVHN + Soccernet              | Soccernet         | 98%                       | 93%                      | 71%                           |
