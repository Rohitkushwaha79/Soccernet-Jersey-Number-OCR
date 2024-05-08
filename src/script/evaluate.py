import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from data.soccernet_data_loader import prepare_soccernet_jersey_dataset
from tensorflow.keras import backend as K
from utils.custom_ctc_acuracy import ctc_accuracy

soccernet_test_img_path = r"C:\Users\Harry\Desktop\number_model\preprocessed_jersey_extract\test\images" 
soccernet_test_dataset= prepare_soccernet_jersey_dataset(soccernet_test_img_path , batch_size=128)


print(len(soccernet_test_dataset))

model = load_model(r'C:\Users\Harry\Desktop\number_model\model94.hdf5')

overall_accuracy = ctc_accuracy(model,soccernet_test_dataset )
print(f"Accuracy on Soccernet test dataset:{overall_accuracy*100:.2f}%")