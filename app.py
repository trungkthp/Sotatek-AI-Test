import streamlit as st
import cv2
import numpy as np
import os

# --- 1. CONFIG ---
st.set_page_config(page_title="Sotatek AI Test", layout="wide")

# --- 2. LOAD MODEL (Hàm quan trọng nhất) ---
@st.cache_resource
def load_models():
    # Import bên trong để tránh lỗi Module khi vừa khởi động
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    import easyocr

    base_path = os.path.dirname(os.path.abspath(__file__))
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    
    # Đảm bảo file model_final.pth nằm trong thư mục output/
    model_path = os.path.join(base_path, "output", "model_final.pth")
    if not os.path.exists(model_path):
        st.error(f"Không tìm thấy model tại {model_path}")
        st.stop()

    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.DEVICE = "cpu" 
    return DefaultPredictor(cfg), easyocr.Reader(['en'])

# --- 3. UI SIDEBAR ---
with st.sidebar:
    st.title("🚀 Sotatek AI Dashboard")
    uploaded_file = st.file_uploader("📤 Upload Drawing", type=["jpg", "png", "jpeg"])
    conf_threshold = st.slider("Confidence", 0.0, 1.0, 0.5)

# --- 4. MAIN LOGIC ---
if uploaded_file is not None:
    predictor, reader = load_models()
    
    # Chuyển file thành ảnh OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    with st.spinner('AI đang xử lý...'):
        outputs = predictor(image)
        instances = outputs["instances"].to("cpu")
        # ... (Phần vẽ Box và JSON giữ nguyên như cũ)
        st.success("Đã phân tích xong!")
        st.json({"status": "success", "count": len(instances)})
else:
    st.info("Hãy tải ảnh lên để bắt đầu.")
