import cv2
import numpy as np

def correct_rotation(image, detected_class):
    """
    Corrects the rotation of the ID image based on the detected class.

    Args:
        image (numpy.ndarray): The cropped ID image.
        detected_class (float): The class of the detected ID:
            - 0.0: No rotation
            - 1.0: Rotated left (90° counter-clockwise)
            - 2.0: Rotated down (180°)
            - 3.0: Rotated right (90° clockwise)

    Returns:
        numpy.ndarray: The corrected and aligned ID image.
    """
    if detected_class == 0.0:
        return image  # No rotation needed

    # Rotate based on the detected class
    if detected_class == 1.0:
        # Rotate left (90° counter-clockwise)
        corrected_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)        
    elif detected_class == 2.0:
        # Rotate down (180°)
        corrected_image = cv2.rotate(image, cv2.ROTATE_180)
    elif detected_class == 3.0:
        # Rotate right (90° clockwise)
        corrected_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        # If an invalid class is provided, return the original image
        print(f"Invalid class detected: {detected_class}. No rotation applied.")
        return image
    cv2.imwrite('data/output/corrected_id.jpg', corrected_image)
    return corrected_image


def crop_ID_image(image, bbox):
    """
    Crops a region from the image using the bounding box.

    Args:
        image (numpy.ndarray): Input image.
        bbox (list or numpy.ndarray): Bounding box [x_min, y_min, x_max, y_max].

    Returns:
        numpy.ndarray: Cropped region of the image.
    """
    x_min, y_min, x_max, y_max = bbox

    # Ensure the bounding box coordinates are within the image dimensions
    x_min = max(0, x_min)
    y_min = max(0, y_min)

    x_max = min(image.shape[1], x_max)
    y_max = min(image.shape[0], y_max)

    # Crop the region from the image
    cropped = image[y_min:y_max, x_min:x_max]

    #save the cropped image
    cv2.imwrite('data/output/cropped_id.jpg', cropped)

    return cropped

def crop_ID_parts(image, parts):
    """
    Crop the parts of the ID using the provided bounding boxes.

    Args:
        image (numpy.ndarray): Input image.
        parts (dict): Dictionary containing part names as keys and bounding boxes as values.

    Returns:
        dict: Dictionary containing part names as keys and cropped images as values.
    """
    # {'Birth': [113, 787, 609, 947], 'Address': [920, 533, 1617, 712], 'Name': [875, 309, 1616, 512], 'Factory': [154, 973, 600, 1072], 'ID_F': [760, 820, 1639, 967]}
    # Crop each part using the provided bounding box

    cropped_parts = {}
    for part_name, bbox in parts.items():
        x_min, y_min, x_max, y_max = bbox
        
        # Ensure coordinates are within image bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(image.shape[1], x_max)
        y_max = min(image.shape[0], y_max)
        
        # Perform the crop
        cropped = image[y_min:y_max, x_min:x_max]
        cropped_parts[part_name] = cropped
        
        # Save cropped part
        output_path = f'data/output/{part_name}.jpg'
        cv2.imwrite(output_path, cropped)

    return None