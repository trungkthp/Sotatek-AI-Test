# **🛠️ Engineering Drawing Analysis: Detection & OCR Pipeline 🏛️**

A comprehensive AI-powered system designed to automate the extraction of information from technical engineering drawings. This project utilizes Faster R-CNN (via Detectron2) for object detection and EasyOCR for intelligent text recognition, all integrated into a seamless Streamlit dashboard.

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/trungkthp/sotatek-ai-test)

> 💡 **Web Demo:** [Click vào đây để trải nghiệm](https://trungkthp-drawing-analysis.hf.space/)
---

## **I. Overview**

* This system addresses the challenge of manual data entry in mechanical engineering by providing:
* **Object Detection**: High-accuracy identification of components such as PartDrawings, Notes, and Tables.
* **Intelligent OCR**: Automated text extraction from localized regions of interest.
* **Heuristic Post-processing**: A custom logic layer that corrects labels based on text density and confidence scores to ensure data integrity.

---

## **II. System Architecture**

The pipeline follows a modular flow:

1. **Input**: Technical drawing upload.
2. **Inference**: Detectron2 identifies bounding boxes for various entities.
3. **Cropping**: Targeted regions are cropped for localized analysis.
4. **OCR Engine**: EasyOCR processes the crops to extract alphanumeric data.
5. **Refinement**: Post-processing scripts validate and correct labels.
6. **Output**: Visual results on the dashboard and structured data export.

---

## **III. Project Structure**

```text
Sotatek-AI-Test/
├── 🖥️ app.py                # Main Streamlit application (UI & Logic)
├── 📁 output/               # Model weights and inference artifacts
│   └── 📄 model_final.pth    # Pre-trained Faster R-CNN weights
├── 📄 test.py               # Model validation script
├── 📄 final_ocr.py          # Advanced OCR processing logic
├── 📄 train.py              # Training pipeline and configuration
├── 📄 requirements.txt      # Python dependencies
├── 📄 .gitignore            # Excludes large binaries and cache files
└── 📚 README.md             # Project documentation
```

## **IV. Technology Stack**
- AI/ML Frameworks: Detectron2 (Facebook Research), PyTorch, EasyOCR.

- Frontend: Streamlit for real-time interactive dashboards.

- Computer Vision: OpenCV-Python, Pillow.

- Data Handling: NumPy, Pandas.

## **V. Quick Start Guide**
**1. Prerequisites**
- OS: Ubuntu 20.04+ (Recommended).

- Python: Version 3.8 hoặc cao hơn.

- Hardware: 8GB+ RAM (Khuyến khích sử dụng GPU).

**2. Installation**


**Clone the repository:**
```
git clone https://github.com/trungkthp/Sotatek-AI-Test.git
cd Sotatek-AI-Test
```

**Install dependencies:**
```
pip install -r requirements.txt --break-system-packages
pip install 'git+https://github.com/facebookresearch/detectron2.git' --break-system-packages
```
**3. Model Weights Setup**

- Do giới hạn kích thước tệp của GitHub, trọng số mô hình (model_final.pth) được lưu trữ bên ngoài.


- Download Link: [Google Drive Link](https://drive.google.com/file/d/1voIzijFduwDECvD2OW_OUr4WONvWz9tW/view?usp=sharing)


- Placement: Tải xuống và di chuyển tệp vào thư mục output/.

