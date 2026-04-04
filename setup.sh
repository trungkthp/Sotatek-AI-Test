#!/bin/bash
echo "Installing base requirements..."
pip install -r requirements.txt --break-system-packages

echo "Installing Detectron2 (Force)..."
pip install 'git+https://github.com/facebookresearch/detectron2.git' --break-system-packages

echo "Done! Run the app with: python3 -m streamlit run app.py"
