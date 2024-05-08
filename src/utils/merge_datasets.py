import tensorflow as tf
from data.svhn_data_loader import process_single_sample

def merge_datasets(dataset1, dataset2 , batch_size):
    """Merges two TensorFlow datasets while preserving label information.

    Args:
        train_dataset: A TensorFlow dataset representing the training data.
        test_dataset: A TensorFlow dataset representing the testing data.

    Returns:
        A merged TensorFlow dataset containing both training and testing data.
    """

    # Ensure both datasets have the same structure (images, labels)
    train_element = next(iter(dataset1))
    test_element = next(iter(dataset2))

    if len(train_element) != len(test_element):
        raise ValueError("Datasets must have the same structure (image, label)")

    # Create a new dataset from the concatenation of train and test datasets
    merged_dataset = dataset1.concatenate(dataset2)
    merged_dataset = (merged_dataset
                    .map(process_single_sample)
                    .batch(batch_size)
                    .prefetch(tf.data.AUTOTUNE))

    return merged_dataset
