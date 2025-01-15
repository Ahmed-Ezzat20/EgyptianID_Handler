# **Lines Extraction from ID Documents**

## **Description**
This project extracts text lines from scanned Egyptian ID documents, correcting alignment and segmenting text lines using YOLOv8, OpenCV.

### **Features**
- **ID Card Detection**: Detects ID cards in images and identifies their orientation using a YOLOv8 model, ensuring accurate detection regardless of the card's position.
- **Rotation Correction**: Corrects misaligned ID cards based on the detected orientation using the Hough Transform, ensuring the card is properly aligned for further processing.
- **Line Segmentation**: Segments ID card images into individual text lines using OpenCV, allowing for precise extraction of each line of text.


---

## **Requirements**
The following Python libraries are required to run the project:
```bash
numpy
opencv-python
Pillow
ultralytics
```
Install all dependencies using the following command:
```bash
pip install -r requirements.txt
```

---

## **Project Structure**
```
lines_extraction_project/
├── data/
│   ├── input/              # Input ID images
│   ├── output/             # Segmented line images and final results
├── models/
│   ├── id_detector/        # YOLOv8 model for ID detection
│   ├── part_detector/      # YOLOv8 model for part detection
├── scripts/
│   ├── detect_id.py        # Script to detect the ID card
│   ├── detect_parts.py     # Script to detect parts within the ID
│   ├── main_pipeline.py    # Main script integrating all steps
├── utils/
│   ├── image_utils.py      # Helper functions for image processing
├── requirements.txt        # Required Python libraries
├── README.md               # Documentation
```

---

## **Usage**
Follow these steps to run the code and reproduce the results:

### **1. Clone the Repository**
Clone the repository to your local machine:
```bash
git clone https://github.com/Ahmed-Ezzat20/EgyptianID_Handler.git
cd EgyptianID_Handler
```

### **2. Place Input Images**
Place the scanned ID images in the `data/input/` directory. Example:
```
data/input/ID.jpg
```

### **3. Run the Pipeline**
Execute the main pipeline script with the following command:
```bash
python scripts/main_pipeline.py --image_path data/input/front.jpg
```

### **4. Outputs**
- **Cropped Lines**: The segmented text lines are saved in the `data/output/` directory.