from collections import Counter

def predict_majority_jersey_no_for_playerid(player_id, pred_label_list):
    """
    Predicts the majority jersey number for a player based on a list of labels.

    Args:
        player_id (int): The id number associated with the player.
        pred_label_list (list): A list of class labels.

    Returns:
        dict: A dictionary containing the player_id and the predicted label.
    """
    # Count occurrences of each label
    counts = Counter(pred_label_list)
    total_count = len(pred_label_list)
    
    # Calculate the percentage of each label
    percentages = {key: (count / total_count) * 100 for key, count in counts.items()}
    
    # Sort labels by percentage in descending order
    sorted_percentage = sorted(percentages.items(), key=lambda x: x[1], reverse=True)
    
    # Get the top two elements with the highest percentage
    top_two = sorted_percentage[:2]
    
    # Extract keys and values from the top two elements
    keys = [item[0] for item in top_two]
    values = [item[1] for item in top_two]
    
    # Check conditions to determine the predicted label
    if values[0] > 99.5 and keys[0] == "-1":
        pred_label = int(keys[0])  # If "-1" has a very high percentage, choose it
    elif int(values[0]) > int(values[1]) and keys[0] != "-1":
        pred_label = int(keys[0])  # Choose the label with the highest percentage if it's not "-1"
    else:
        pred_label = int(keys[1])  # Otherwise, choose the second label
    
    # Create a dictionary with the predicted label for the player
    pred_dict = {str(player_id): pred_label}
    
    return pred_dict
