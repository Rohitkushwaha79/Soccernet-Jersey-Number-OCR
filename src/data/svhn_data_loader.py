import pandas as pd
import numpy as np
import mat73
import os
import cv2
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences



def load_bbox_data_from_mat(mat_path):
    """
    Load bounding box data from a .mat file.

    Args:
    - mat_path: Path to the .mat file.

    Returns:
    - Dictionary containing bounding box data.
    """
    bbox_data = mat73.loadmat(mat_path)
    return bbox_data


def convert_to_list(value):
    """
    Convert a value to a list if it is not already a list.

    Args:
    - value: Value to be converted.

    Returns:
    - List containing the value.
    """
    if not isinstance(value, list):
        return [value]
    return value

def create_dataframe_from_mat_data(mat_data, image_path):
    """
    Creates a DataFrame from MAT data containing bounding box information.

    Args:
    - mat_data: Dictionary containing bounding box information extracted from a .mat file.
    - image_path: Path to the directory containing images.

    Returns:
    - DataFrame containing bounding box information and image paths.
    """
    char_list =['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    height = []
    labels = []
    left = []
    top = []
    width = []
    img_name = [os.path.join(image_path, name) for name in mat_data["digitStruct"]["name"]]
    for item in mat_data["digitStruct"]["bbox"]:
        height.append(item['height'])
        labels.append(item['label'])
        left.append(item['left'])
        top.append(item['top'])
        width.append(item['width'])
    labels = [convert_to_list(item) for item in labels]
    max_label_len = max([len(text) for text in labels])
    height = [convert_to_list(item) for item in height]
    width = [convert_to_list(item) for item in width]
    top = [convert_to_list(item) for item in top]
    left = [convert_to_list(item) for item in left]
    labels = [[int(item)%10 for item in label] for label in labels]
    labels = [pad_sequences([label], maxlen=6, padding='post' ,value = len(char_list))[0] for label in labels]
    dataframe = pd.DataFrame({"img_path": img_name, 'height': height, 'label': labels, 'left': left, 'top': top, 'width': width})
    return dataframe






def create_dataframe_from_svhn_data(mat_path, img_path):
    """
    Creates a DataFrame from SVHN dataset information stored in .mat file and image directory.

    Args:
    - mat_path: Path to the .mat file containing SVHN dataset information.
    - img_path: Path to the directory containing SVHN images.

    Returns:
    - DataFrame containing SVHN dataset information.
    """
    mat_data = load_bbox_data_from_mat(mat_path)
    svhn_df = create_dataframe_from_mat_data(mat_data, img_path)
    return svhn_df



def preprocess_and_extract_roi_from_row(row):
    """
    Preprocesses and extracts regions of interest (ROI) from an image represented by a DataFrame row.

    Args:
    - row: DataFrame row containing image and ROI information.

    Returns:
    - List of preprocessed ROI images.
    - List of corresponding labels.
    """
    img = cv2.imread(row['img_path'])
    if img is None:
        return None, None

    top_list = row['top']
    left_list = row['left']
    height_list = row['height']
    width_list = row['width']
    label_list = row['label']

    concatenated_roi = None
    labels = []
    preprocessed_images = []

    common_height = 32  # Set a common height for all ROIs

    for i in range(len(top_list)):
        top = abs(top_list[i])
        left = abs(left_list[i])
        height = abs(height_list[i])
        width = abs(width_list[i])

        object_roi = img[int(top):int(top + height), int(left):int(left + width)]

        # Resize each ROI to the common height
        resized_object_roi = cv2.resize(object_roi, (int(width / height * common_height), common_height))

        if concatenated_roi is None:
            concatenated_roi = resized_object_roi
        else:
            concatenated_roi = np.concatenate((concatenated_roi, resized_object_roi), axis=1)

    labels.append(label_list)

    resized_concatenated_roi = cv2.resize(concatenated_roi, (64, 32))
    grayscale_concatenated_roi = cv2.cvtColor(resized_concatenated_roi, cv2.COLOR_BGR2GRAY)

    preprocessed_images.append(grayscale_concatenated_roi)

    return preprocessed_images, labels




def preprocess_and_extract_roi_from_dataframe(dataframe):
    """
    Preprocesses and extracts regions of interest (ROI) from images in the given DataFrame.

    Args:
    - dataframe: DataFrame containing image information.

    Returns:
    - List of preprocessed images.
    - List of corresponding labels.
    """
    preprocessed_images = []
    labels = []

    for index, row in dataframe.iterrows():
        images, lbls = preprocess_and_extract_roi_from_row(row)
        if images is not None and lbls is not None:
            preprocessed_images.extend(images)
            labels.extend(lbls)

    return preprocessed_images, labels



def create_svhn_dataset(mat_path, img_path):
    """
    Creates an SVHN dataset from provided .mat file and image directory.

    Args:
    - mat_path: Path to the .mat file containing SVHN dataset information.
    - img_path: Path to the directory containing SVHN images.

    Returns:
    - TensorFlow dataset containing preprocessed SVHN images and labels.
    """
    dataframe = create_dataframe_from_svhn_data(mat_path, img_path)
    preprocessed_images, labels = preprocess_and_extract_roi_from_dataframe(dataframe)
    preprocessed_images = np.array(preprocessed_images)
    labels = np.array(labels)
    dataset = tf.data.Dataset.from_tensor_slices((preprocessed_images, labels))
    return dataset


def process_single_sample(img, label):
    return {"image": img, "label": label}

# preprocessing  dataset 
def prepare_svhn_dataset(mat_path, img_path, batch_size):
    """
    Prepares the SVHN dataset for training or evaluation.

    Args:
    - mat_path: Path to the .mat file containing SVHN dataset information.
    - img_path: Path to the directory containing SVHN images.
    - batch_size: Batch size for batching the dataset.

    Returns:
    - Prepared dataset ready for training or evaluation.
    """
    dataset = create_svhn_dataset(mat_path, img_path)
    dataset = (dataset
               .map(process_single_sample)
               .batch(batch_size)
               .prefetch(tf.data.AUTOTUNE))

    return dataset


