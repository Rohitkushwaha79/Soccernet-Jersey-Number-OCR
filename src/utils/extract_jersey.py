import cv2
import numpy as np


def extract_jersey_from_frame(image_path , pose_detection_model):
    """
    Extracts the region containing the jersey number from a soccer player frame image.

    Args:
    - image_path (str): Path to the input image file.

    Returns:
    - img1 (numpy.ndarray): Cropped and resized image containing the jersey number region.
                            Returns None if jersey number extraction fails.
    """
    # Read the image
    image = cv2.imread(image_path)

    # Check if image is None
    if image is None:
        return None

    # Resize image to a fixed size
    image = cv2.resize(image, (285, 600))
    im_height, im_width = image.shape[:2]

    # Perform pose estimation using YOLO model
    results = pose_detection_model(image)
    keypoints = results[0].keypoints

    # Set confidence threshold for keypoints
    threshold = 0.1
    points = []

    # Extract keypoints with confidence above threshold
    for kpt_data in keypoints.data[0]:
        prob = kpt_data[2].item()
        if prob > threshold:
            x = kpt_data[0].item()
            y = kpt_data[1].item()
            points.append((int(x), int(y)))
        else:
            points.append(None)

    # Check if keypoints are visible
    if keypoints.has_visible is False:
        points = [None] * 13

    # Define quadrilateral coordinates for jersey region
    quadrilateral_coords = [points[6], points[5], points[12], points[11]]

    # Check if any of the points are None
    if None in quadrilateral_coords:
        return None

    # Calculate bounding box coordinates
    x1, y1 = points[5]
    x2, y2 = points[6]
    x3, y3 = points[11]
    x4, y4 = points[12]
    top = np.abs(min(y1, y2, y3, y4))
    left = np.abs(min(x1, x2, x3, x4))
    height = np.abs(max(y1, y2, y3, y4) - top)
    width = np.abs(max(x1, x2, x3, x4) - left)

    # Check if top or left coordinates are beyond image boundaries
    if top >= im_height * 0.5 or left >= im_width * 0.7:
        return None

    # Check if any coordinate is zero
    if 0 in [top, height, width, left]:
        return None

    # Crop and resize jersey region
    img1 = image[int(top + height * 0.1 - 3):int(top + height * 0.8 + 3), int(left - 10):int(left + width + 10)]
    try:
        img1 = cv2.resize(img1, (64, 32))
    except:
        return None

    # Check for invalid jersey region
    if x1 >= x2 or x1 >= x4 or x3 >= x4 or x3 >= x2 or y1 >= y3 or y1 >= y4 or y2 >= y3 or y2 >= y4 or np.abs(x2 - x1) < 55 or np.abs(x1 - x2) < 55:
        return None
    else:
        return img1