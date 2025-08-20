# Paddle_YOLO_integration
# Expression Detection and Line Separation OCR

This project is designed to build a custom OCR pipeline for detecting and recognizing text expressions from images.  
It uses **YOLO** for detecting expressions, crops them, and then prepares the data for further OCR processing with **PaddleOCR**.  

The pipeline is currently divided into two stages:  
1. **Expression Detection Model (Completed)** – Detects and crops expressions from images.  
2. **Line Separation Model (In Progress)** – Separates cropped text into lines before passing them to PaddleOCR.

---

## Features
- Expression detection using YOLO
- Automatic cropping of detected regions
- Support for integration with PaddleOCR
- Modular design for easy extension

---

## Requirements
- Python 3.9+
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- OpenCV
- PaddleOCR
- PyTorch
- Numpy

Install dependencies:
```bash
pip install ultralytics opencv-python paddleocr torch numpy
