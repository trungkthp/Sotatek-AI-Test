**🛠️ Engineering Drawing Analysis: Detection \& OCR Pipeline 🏛️**
A comprehensive AI-powered system designed to automate the extraction of information from technical engineering drawings. This project utilizes Faster R-CNN (via Detectron2) for object detection and EasyOCR for intelligent text recognition, all integrated into a seamless Streamlit dashboard.

**I. Overview**

* This system addresses the challenge of manual data entry in mechanical engineering by providing:
* Object Detection: High-accuracy identification of components such as PartDrawings, Notes, and Tables.
* Intelligent OCR: Automated text extraction from localized regions of interest.
* Heuristic Post-processing: A custom logic layer that corrects labels based on text density and confidence scores to ensure data integrity.

**II. System Architecture**

The pipeline follows a modular flow:

* Input: Technical drawing upload.
* Inference: Detectron2 identifies bounding boxes for various entities.
* Cropping: Targeted regions are cropped for localized analysis.
* OCR Engine: EasyOCR processes the crops to extract alphanumeric data.
* Refinement: Post-processing scripts validate and correct labels (e.g., distinguishing between a small Table and a large Note).
* Output: Visual results on the dashboard and structured data export.

**III. Project Structure**

Sotatek-AI-Test/
│
├── 🖥️ app.py                # Main Streamlit application (UI \& Logic)
├── 📁 output/               # Model weights and inference artifacts
## III. Project Structure

```text
Sotatek-AI-Test/
├── 🖥️ app.py                # Main Streamlit application (UI & Logic)
├── 📁 output/               # Model weights and inference artifacts
│   └── model_final.pth      # Pre-trained Faster R-CNN weights
├── 📁 src/                  # Core processing modules
│   ├── test.py              # Model validation script
│   ├── final_ocr.py         # Advanced OCR processing logic
│   └── train.py             # Training pipeline and configuration
├── 📁 dataset/              # Training/Validation data (Excluded from Git)
├── 📄 requirements.txt      # Python dependencies
├── 📄 .gitignore            # Excludes large binaries and cache files
└── 📚 README.md             # Project documentation│   └── model\_final.pth      # Pre-trained Faster R-CNN weights
**IV. Technology Stack**

* AI/ML Frameworks: Detectron2 (Facebook Research), PyTorch, EasyOCR.
* Frontend: Streamlit for real-time interactive dashboards.
* Computer Vision: OpenCV-Python, Pillow.
* Data Handling: NumPy, Pandas (for table data structuring).

**V. Quick Start Guide**

1. **Prerequisites**
OS: Ubuntu 20.04+ (Recommended) or Windows with C++ Build Tools.

Python: version 3.8 or higher.

Hardware: 8GB+ RAM (GPU recommended for faster inference).

2. **Installation**

**# Clone the repository**

git clone https://github.com/trungkthp/Sotatek-AI-Test.git
cd Sotatek-AI-Test

**# Set up virtual environment**

python3 -m venv venv
source venv/bin/activate

**# Install dependencies**

pip install -r requirements.txt
pip install 'git+https://github.com/facebookresearch/detectron2.git'

3. **Model Weights Setup**
Due to GitHub's file size limits, the pre-trained weights (model\_final.pth) are hosted externally.
* Download Link: https://drive.google.com/drive/folders/1TTVcZm9NxX8nwmYFx3JskfVWY6wWngoi?usp=drive\_link
* Placement: Download and move the file into the output/ directory.

