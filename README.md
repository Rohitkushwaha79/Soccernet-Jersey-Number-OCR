# Soccernet-Jersey-Number-OCR
## Overview
This repository presents experiments conducted for Optical Character Recognition (OCR) of jersey numbers in soccer images. The experiments involve training and evaluating various models using datasets from both SVHN and SoccerNet Jersey Number Recognition. While models were trained on both datasets, the evaluation focused on the SoccerNet Jersey Number Recognition dataset. The performance of these models was assessed on the SoccerNet Jersey Number Recognition test and challenge datasets, with results reported accordingly.

## Getting Started

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Rohitkushwaha79/Soccernet-Jersey-Number-OCR.git

2. **Install Dependencies**
   * Create a virtual environment to manage project dependencies effectively (recommended).
   * Install the required libraries using pip install -r requirements.txt.
   * Ensure you have TensorFlow, Keras, and other necessary libraries listed in requirements.txt.
  
## Experiments

### Experiment 1: CTC CRNN Model on SVHN Dataset
- **Description:** Trained a CTC (Connectionist Temporal Classification) CRNN (Convolutional Recurrent Neural Network) model on the Street View House Numbers (SVHN) dataset.
- **Train Dataset Used:** SVHN dataset
- **Test Dataset Used:** Soccernet test dataset
- **Accuracy on Test Dataset:** 39%

### Experiment 2: CNN Model on Soccernet Dataset
- **Description:** Trained a Convolutional Neural Network (CNN) model on the Soccernet training dataset.
- **Train Dataset Used:** Soccernet training dataset
- **Test Dataset Used:** Soccernet test dataset
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
    4. Merging SVHN and Soccernet data.
  - **Accuracy on Training Dataset:** 95%
  - **Accuracy on Test Dataset:** 91%
  - **Accuracy on Challenge Dataset:** 69%

## Experiment Results

| Model        | Train Dataset Used           | Test Dataset Used | Accuracy on Soccernet Train Dataset | Accuracy on Soccernet Test Dataset | Accuracy on Challenge Dataset |
|--------------|-------------------------------|-------------------|---------------------------|--------------------------|-------------------------------|
| CTC CRNN     | SVHN                          | Soccernet         | 39%                       | -                        | -                             |
| CNN          | Soccernet                     | Soccernet         | -                         | 24%                      | -                             |
| EasyOCR      | -                             | Soccernet Challenge | -                       | -                        | 58%                           |
| CTC CRNN     | SVHN + Soccernet              | Soccernet         | 98%                       | 93%                      | 69%                           |
