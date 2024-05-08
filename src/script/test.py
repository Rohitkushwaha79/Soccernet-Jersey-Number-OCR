import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from utils.custom_ctc_acuracy import ctc_decoder
from utils.extract_jersey import extract_jersey_from_frame
from ultralytics import YOLO
from tensorflow.keras import backend as K

prediction_model = load_model(r'C:\Users\Harry\Desktop\number_model\model94.hdf5')
pose_detection_model = YOLO(r"C:\Users\Harry\Desktop\number_model\yolov8m-pose.pt" , verbose=False)



player_frame_path = r'C:\Users\Harry\Desktop\number_model\3_10.jpg'
jersey = extract_jersey_from_frame(player_frame_path , pose_detection_model)

if jersey is not None:
    jersey = cv2.resize(jersey, (64, 32))
    gray_jersey = cv2.cvtColor(jersey ,  cv2.COLOR_BGR2GRAY)  
    resized_gray_jersey = np.expand_dims(gray_jersey, axis=0)  


    predictions = prediction_model.predict(resized_gray_jersey)

    pred_jersey_num = ctc_decoder(predictions)

    print("Predicted Jersey Number:", pred_jersey_num)