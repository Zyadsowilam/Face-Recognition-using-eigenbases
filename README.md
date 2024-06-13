### Face Recognition using eigen bases README

#### Overview
This Python script performs face recognition using Principal Component Analysis (PCA) on a dataset of grayscale images. It utilizes eigenfaces to project images into a reduced-dimensional space and compares them using cosine similarity to recognize faces.

#### Requirements
- Python 3.x
- Libraries: `numpy`, `PIL`, `os`, `tkinter`

#### Installation
1. Install Python from [python.org](https://www.python.org/downloads/).
2. Install required libraries:
   ```
   pip install numpy Pillow
   ```
   
#### Usage
- Ensure Python and required libraries are installed.
- Run the script in a Python environment.

#### Functionality
1. **Face Recognition Function**
   - `face_recognition(test_image_path, q=35, threshold=0.7, var_thresholds=[85, 95])`
     - Inputs:
       - `test_image_path`: Path to the test image for recognition.
       - `q`: Number of principal components to consider.
       - `threshold`: Minimum cosine similarity for recognition.
     - Outputs:
       - `predicted_label`: Recognized label of the person in the test image.
       - `predicted_image_path`: Path to the predicted image of the recognized person.

2. **Graphical User Interface (GUI)**
   - Uses `tkinter` for a simple interface to select a test image, set parameters (`q` and `threshold`), and display the original and predicted images.
   - Run the GUI:
     ```
     python face_recognition_gui.py
     ```

#### Example
```python
# Example usage of face_recognition function
test_image_path = r"D:\imgFR\test\s8\7.pgm"
predicted_label, predicted_image_path = face_recognition(test_image_path, q=35, threshold=0.7)
print("Predicted Label:", predicted_label)
```

#### File Structure
- `archive/`: Directory containing training images organized by subject.
- `test/`: Directory containing test images for recognition.

#### Notes
- Ensure the correct path (`folder_path` and `TestFolder`) is set for loading images.
- Adjust `var_thresholds` for different levels of explained variance.

#### Credits
- This script is inspired by concepts from machine learning and image processing literature.


---

This README provides an overview of the face recognition script, its usage, and necessary setup instructions for users and developers. Adjust paths and parameters according to your dataset and requirements.
