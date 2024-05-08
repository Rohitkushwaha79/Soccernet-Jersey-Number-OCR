import pandas as pd
import tensorflow as tf
import numpy as np
import cv2
import os


from data.svhn_data_loader import create_svhn_dataset
from data.soccernet_data_loader import create_soccernet_jersey_dataset
from models.model import get_model 
from utils.merge_datasets import merge_datasets

# svhn data path
svhn_train_img_path = r"C:\Users\Harry\Desktop\number_model\SVHN\train\train"
svhn_test_img_path = r"C:\Users\Harry\Desktop\number_model\SVHN\test\test"
svhn_train_mat_path = r"C:\Users\Harry\Desktop\number_model\SVHN\train_digitStruct.mat"
svhn_test_mat_path = r"C:\Users\Harry\Desktop\number_model\SVHN\test_digitStruct.mat"

# soccernet data path
soccernet_train_img_path = r"C:\Users\Harry\Desktop\number_model\preprocessed_jersey_extract\train\images"
soccernet_test_img_path = r"C:\Users\Harry\Desktop\number_model\preprocessed_jersey_extract\test\images"

# loading svhn train and test dataset
svhn_train_dataset = create_svhn_dataset(mat_path= svhn_train_mat_path , img_path= svhn_train_img_path)
svhn_test_dataset = create_svhn_dataset(mat_path= svhn_test_mat_path , img_path= svhn_test_img_path)

# loading soccernet train and test dataset
soccernet_train_dataset = create_soccernet_jersey_dataset(img_path=soccernet_train_img_path)
soccernet_test_dataset = create_soccernet_jersey_dataset(img_path=soccernet_test_img_path)

# merging svhn and soccernet datasets
merged_train_dataset = merge_datasets(dataset1=svhn_train_dataset , dataset2=soccernet_train_dataset , batch_size=128)
merged_test_dataset = merge_datasets(dataset1=svhn_test_dataset , dataset2= soccernet_test_dataset , batch_size=128)

# callback
reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1024,
    patience=3,
    verbose=0,)

# model
model = get_model()


history = model.fit(merged_train_dataset,
          epochs=30,
          validation_data=merged_test_dataset,
          callbacks=[reduce_lr_callback])

model.save("model.hdf5")