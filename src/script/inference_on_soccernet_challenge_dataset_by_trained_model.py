import os 
import cv2
import numpy as np
from utils.predict_majority_jersey_no_for_playerid import predict_majority_jersey_no_for_playerid
from utils.extract_jersey import extract_jersey_from_frame
import json
from ultralytics import YOLO
from utils.custom_ctc_acuracy import ctc_decoder
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

prediction_model = load_model(r'C:\Users\Harry\Desktop\number_model\model94.hdf5')
pose_detection_model = YOLO(r"models/yolov8m-pose.pt" , verbose=False)

challenge_path = r"data/SoccerNet/jersey-2023/challenge/challenge/images"
predictions = {}

# Loop through player IDs
for player_id in (os.listdir(challenge_path)):
    pred_label_list = []
    
    # Define the path to the folder containing player frames
    player_frame_folder_path = os.path.join(challenge_path, str(player_id))
    
    # Get paths to each frame for the player
    player_frames_path = [os.path.join(player_frame_folder_path, frame) for frame in os.listdir(player_frame_folder_path)]
    
    # Process each frame
    for frame_path in player_frames_path:
        print(frame_path)
        
        # Extract jersey from the frame
        jersey_extract_roi = extract_jersey_from_frame(frame_path)
        
        if jersey_extract_roi is not None:
            # Resize and convert the extracted jersey to grayscale
            resized_jersey_extract_roi = cv2.resize(jersey_extract_roi, (64,32))  
            gray_jersey_extract_roi = cv2.cvtColor(resized_jersey_extract_roi, cv2.COLOR_BGR2GRAY)
            gray_jersey_extract_roi = new_image = np.expand_dims(gray_jersey_extract_roi, axis=0)  
            # Use EasyOCR to recognize text (jersey numbers) from the grayscale image
            preds = prediction_model.predict(gray_jersey_extract_roi)
            pred_texts = ctc_decoder(preds)
            print(pred_texts)
            if pred_texts==['']:

                pred_label_list.append("-1")
            else:
                pred_label_list.append(pred_texts[0])
        else:
            pred_label_list.append("-1")
    # Predict the majority jersey number for the player
    predictions_dic = predict_majority_jersey_no_for_playerid(player_id, pred_label_list)
    
    # Add the prediction to the overall predictions dictionary
    predictions = {**predictions, **predictions_dic}

print(predictions)

# saving predictions in json format
 
file_path = 'output.json'

# Save dictionary to JSON file
with open(file_path, 'w') as json_file:
    json.dump(predictions, json_file)