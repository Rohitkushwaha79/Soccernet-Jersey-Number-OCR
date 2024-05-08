import os 
import cv2
import easyocr
import cv2
from utils.predict_majority_jersey_no_for_playerid import predict_majority_jersey_no_for_playerid
from utils.extract_jersey import extract_jersey_from_frame
import json
from ultralytics import YOLO

reader = easyocr.Reader(['en'])
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
        jersey_extract_roi = extract_jersey_from_frame(image_path=frame_path , pose_detection_model= pose_detection_model)
        
        if jersey_extract_roi is not None:
            # Resize and convert the extracted jersey to grayscale
            resized_jersey_extract_roi = cv2.resize(jersey_extract_roi, (128, 128))  
            gray_jersey_extract_roi = cv2.cvtColor(resized_jersey_extract_roi, cv2.COLOR_BGR2GRAY)
            
            # Use EasyOCR to recognize text (jersey numbers) from the grayscale image
            predicted_output = reader.readtext(gray_jersey_extract_roi, allowlist='0123456789')
            
            # If text is recognized, append the predicted jersey number to the list
            if len(predicted_output) > 0:
                if predicted_output[0][-2] == '':
                    pred_label_list.append("-1")  # Append "-1" if no text is recognized
                else:
                    pred_label_list.append(predicted_output[0][-2])  # Append recognized jersey number
            else:
                pred_label_list.append("-1")  # Append "-1" if no text is recognized
#         else:
#             pred_label_list.append("-1")  # Append "-1" if no jersey is extracted
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

