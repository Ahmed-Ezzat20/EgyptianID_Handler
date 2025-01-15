
# **Lines Extraction from ID Documents**

## **Description**
This project implements a system to extract individual lines of text as cropped images from scanned Egyptian ID documents, regardless of orientation. The system detects the ID card, corrects its alignment, segments it into text lines, and extracts the text using OCR.

### **Features**
- **ID Card Detection**: Detects ID cards and identifies their orientation using a YOLOv8 model.
- **Rotation Correction**: Corrects misaligned IDs using the detected class and Hough Transform.
- **Line Segmentation**: Segments ID images into individual text lines using OpenCV.
- **OCR**: Extracts text from segmented lines using PyTesseract.

---

## **Dataset**
- The system is trained and tested on Egyptian IDs.
- Users can provide their dataset for additional training or testing.

---

## **Requirements**
The following Python libraries are required to run the project:
```bash
ultralytics
opencv-python
numpy
Pillow
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
git clone <repository_url>
cd lines_extraction_project
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
- **Cropped Lines**: The segmented text lines are saved in the `data/output/` directory.#   E g y p t i a n I D _ H a n d l e r  
 #   E g y p t i a n I D _ H a n d l e r  
 