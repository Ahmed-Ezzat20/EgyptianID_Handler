from ultralytics import YOLO
from utils.image_utils import crop_ID_image, correct_rotation
import cv2


model = YOLO('models/id_detector/best.pt')  # Load the trained model

def detect_id(image_path):
    """
    Detects the ID in the input image and applies rotation correction if necessary.

    Args:
        image_path (str): Path to the input image.

    Returns:
        numpy.ndarray: Corrected and cropped ID image.
    """
    # Load the image
    image = cv2.imread(image_path)

    # Detect objects
    results = model.predict(source=image_path, conf=0.7, save=True)

    # Extract the class ID of the detected object
    detected_class = results[0].boxes.cls[0].item()
    print(f"Detected class: {detected_class}")
    print(f"Class names: {results[0].names}")

    # Handle valid ID classes (rotated or not)
    if detected_class not in [0.0, 1.0, 2.0, 3.0]:
        print("No valid ID detected.")
        return None

    # Extract the bounding box of the first detected ID
    id_bbox = results[0].boxes.xyxy[0].cpu().numpy().astype(int)

    # Crop the detected ID region
    cropped_id = crop_ID_image(image, id_bbox)

    # Skip rotation correction for class 0
    if detected_class == 0.0:
        print("No rotation detected. Skipping rotation correction.")
        return cropped_id

    # Correct rotation for rotated IDs
    corrected_id = correct_rotation(cropped_id, detected_class)

    return corrected_id