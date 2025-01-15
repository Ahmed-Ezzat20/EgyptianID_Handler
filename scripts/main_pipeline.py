import sys
import os
import cv2
import argparse
# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.detect_id import detect_id
from scripts.detect_parts import detect_parts


def main_pipeline(image_path):
    """
    Main pipeline to process the input image, detect ID, extract parts, and perform OCR.

    Args:
        image_path (str): Path to the input image.

    Returns:
        dict: Extracted text from each part of the ID.
    """
    # Step 1: Detect and correct the ID
    cropped_id = detect_id(image_path)

    # Step 2: Detect parts
    parts = detect_parts(cropped_id)

    return None


if __name__ == '__main__':
    # Create argument parser
    parser = argparse.ArgumentParser(description='Extract text from ID documents')
    parser.add_argument('--image_path', type=str, required=True,
                      help='Path to the input ID image')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run pipeline with provided image path
    main_pipeline(args.image_path)