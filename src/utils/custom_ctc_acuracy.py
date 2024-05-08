from itertools import groupby
import tensorflow as tf
import numpy as np

char_list =['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
def ctc_decoder(predictions):
    '''
    input: given batch of predictions from text rec model
    output: return lists of raw extracted text

    '''
    text_list = []
    
    pred_indcies = np.argmax(predictions, axis=-1)
    
    for i in range(pred_indcies.shape[0]):
        ans = ""
        
        ## merge repeats
        merged_list = [k for k,_ in groupby(pred_indcies[i])]
        
        ## remove blanks
        for p in merged_list:
            if p != len(char_list):
                ans += char_list[int(p)]
        
        text_list.append(ans)
        
    return text_list


# Define your ctc_accuracy function
def ctc_accuracy(prediction_model, dataset, batch_size=1000):
    total_correct = 0
    total_samples = 0

    all_images = []
    all_labels = []

    # Iterate through the entire val_dataset to accumulate the images and labels
    for batch in dataset:
        images = batch["image"]  # Access the image tensor for the current batch
        labels = batch["label"]  # Access the label tensor for the current batch

        all_images.append(images)  # Append the images to the list
        all_labels.append(labels)  # Append the labels to the list

    # Convert the lists to NumPy arrays if needed
    all_images = np.concatenate(all_images, axis=0)  # Concatenate the list of images into a single array
    all_labels = np.concatenate(all_labels, axis=0)  # Concatenate the list of labels into a single array
        # Iterate over batches
    for i in range(0, len(all_images), batch_size):
        batch_images = all_images[i:i+batch_size]
        batch_labels = all_labels[i:i+batch_size]

        # Perform prediction on the batch
        y_pred = prediction_model(batch_images)

        # Decode predictions
        pred_text = ctc_decoder(y_pred)

        # Convert labels to strings
        org_text = []
        for labels in batch_labels:
            ans = ''.join([str(label) for label in labels if label != 10])
            org_text.append(ans)

        # Compute accuracy for the batch
        correct_predictions = sum([1 for pred, true in zip(pred_text, org_text) if pred == true])
        total_correct += correct_predictions
        total_samples += len(batch_images)

    # Compute overall accuracy
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    return overall_accuracy




