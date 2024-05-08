import pandas as pd
import numpy as np
import mat73
import os
import cv2
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences




def create_dataframe_from_images(path):
    """
    Creates a DataFrame from images in a directory, extracting labels from directory names.

    Args:
    - path: Path to the directory containing images.

    Returns:
    - DataFrame containing image paths and corresponding labels.
    """
    labels = []
    paths = []
    for directory_name in os.listdir(path):
        directory_path = os.path.join(path, directory_name)
        if os.path.isdir(directory_path):
            for filename in os.listdir(directory_path):
                file_path = os.path.join(directory_path, filename)
                if str(directory_name) == "-1":
                    labels.append([10])
                else:
                    labels.append([int(digit) for digit in str(directory_name)])
                paths.append(file_path)
        else:
            print("Not a directory:", directory_path)

    labels = [pad_sequences([label], maxlen=6, padding='post', value=10)[0] for label in labels]
    return pd.DataFrame({"Image_Path": paths, "Labels": labels})




def process_images_to_dataframe(img_path):
    """
    Processes images in a directory and creates a DataFrame.

    Args:
    - img_path: Path to the directory containing images.

    Returns:
    - DataFrame containing processed image data.
    """
    dataframe = create_dataframe_from_images(img_path)
    return dataframe


def extract_roi_from_dataframe(row):
    """
    Extracts region of interest (ROI) from an image and its corresponding label from a DataFrame row.

    Args:
    - row: DataFrame row containing image path and label information.

    Returns:
    - Tuple containing preprocessed images and labels.
    """
    img = cv2.imread(row['Image_Path'])
    if img is None:
        return None, None

    label_list = row['Labels']
    labels = []
    preprocessed_images = []

    labels.append(label_list)

    resized_img = cv2.resize(img, (64, 32))
    grayscale_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

    preprocessed_images.append(grayscale_img)

    return preprocessed_images, labels




def extract_roi_batch_from_dataframe(dataframe):
    """
    Extracts regions of interest (ROIs) from images and their corresponding labels from a DataFrame.

    Args:
    - dataframe: DataFrame containing image paths and label information.

    Returns:
    - Tuple containing preprocessed images and labels for the entire DataFrame.
    """
    preprocessed_images = []
    labels = []

    for index, row in dataframe.iterrows():
        images, lbls = extract_roi_from_dataframe(row)
        if images is not None and lbls is not None:
            preprocessed_images.extend(images)
            labels.extend(lbls)

    return preprocessed_images, labels




def create_soccernet_jersey_dataset(img_path):
    """
    Creates a TensorFlow dataset from the raw data of SoccerNet jersey images.

    Args:
    - img_path: Path to the directory containing SoccerNet jersey images.

    Returns:
    - TensorFlow dataset containing preprocessed images and labels.
    """
    dataframe = process_images_to_dataframe(img_path)

    preprocessed_images, labels = extract_roi_batch_from_dataframe(dataframe)
    preprocessed_images = np.array(preprocessed_images)
    labels = np.array(labels)

    dataset = tf.data.Dataset.from_tensor_slices((preprocessed_images, labels))

    return dataset




def process_single_sample(img, label):
    return {"image": img, "label": label}


def prepare_soccernet_jersey_dataset(img_path, batch_size):
    """
    Prepares the SoccerNet jersey dataset for training or evaluation.

    Args:
    - img_path: Path to the directory containing SoccerNet jersey images.
    - batch_size: Batch size for batching the dataset.

    Returns:
    - Prepared dataset ready for training or evaluation.
    """
    dataset = create_soccernet_jersey_dataset(img_path)
    dataset = (dataset
               .map(process_single_sample)
               .batch(batch_size)
               .prefetch(tf.data.AUTOTUNE))

    return dataset






