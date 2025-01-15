import cv2
from ultralytics import YOLO
from utils.image_utils import crop_ID_parts

def detect_parts(cropped_id):
    # Load YOLO model
    model = YOLO('models/part_detector/best.pt')
    
    # Define part names mapping (adjust based on your model's classes)
    part_names = {0: 'Address', 1: 'Birth', 2: 'Factory', 3: 'Gender', 4: 'HusName', 5: 'ID_B', 6: 'ID_F', 7: 'Name', 8: 'Occup', 9: 'Rel', 10: 'Start', 11: 'Status', 12: 'end'}
    
    results = model.predict(cropped_id, save=True)
    parts = {}
    
    # Process first image results (assuming single image input)
    if len(results) > 0:
        boxes = results[0].boxes
        
        # Iterate through each detection
        for i in range(len(boxes.xyxy)):
            # Get coordinates
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            # Get class ID
            class_id = int(boxes.cls[i].item())
            # Get confidence
            conf = boxes.conf[i].item()
            
            # Only keep detections above confidence threshold
            if conf > 0.5:  # Adjust threshold as needed
                part_name = part_names.get(class_id, f"unknown_{class_id}")
                parts[part_name] = [int(x1), int(y1), int(x2), int(y2)]
    
    print(parts)
    crop_ID_parts(cropped_id, parts)
    return None
