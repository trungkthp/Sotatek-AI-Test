import streamlit as st
import cv2
import numpy as np
import os
import time
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import easyocr

# --- IMPORT MODULE TỪ THƯ MỤC SRC ---
# Nếu bạn có logic OCR phức tạp trong final_ocr.py, hãy import nó ở đây
# from src import final_ocr 

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="Sotatek AI Drawing Dashboard", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# CSS để Dashboard nhìn "sang" hơn
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stAlert { border-radius: 10px; }
    .st-expander { border: 1px solid #e0e0e0; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOAD MODELS (Đã sửa đường dẫn để khớp với GitHub) ---
@st.cache_resource
def load_models():
    base_path = os.path.dirname(os.path.abspath(__file__))
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    
    # Cách kiểm tra linh hoạt:
    path_in_output = os.path.join(base_path, "output", "model_final.pth")
    path_in_root = os.path.join(base_path, "model_final.pth")
    
    if os.path.exists(path_in_output):
        model_path = path_in_output
    elif os.path.exists(path_in_root):
        model_path = path_in_root
    else:
        st.error(f"❌ Không tìm thấy model tại {path_in_output} hoặc {path_in_root}")
        st.stop()
        
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.DEVICE = "cpu" 
    
    return DefaultPredictor(cfg), easyocr.Reader(['en'])

# --- 3. SIDEBAR ---
with st.sidebar:
    # Thay logo bằng text hoặc ảnh local nếu link die
    st.title("🚀 Sotatek AI Test")
    st.subheader("Technical Drawing Analysis")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("📤 Upload Engineering Drawing", type=["jpg", "png", "jpeg"])
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.50)
    
    st.markdown("---")
    st.write("📌 **Project Information:**")
    st.caption("- Model: Faster R-CNN (ResNet-101)")
    st.caption("- Framework: Detectron2 & EasyOCR")
    st.write("👤 Candidate: **Trung - AI Engineer**")

# --- 4. XỬ LÝ CHÍNH ---
if uploaded_file is not None:
    predictor, reader = load_models()
    class_names = ["PartDrawing", "Note", "Table"]
    # Màu sắc chuyên nghiệp (RGB)
    colors = {"PartDrawing": (255, 0, 0), "Note": (0, 150, 0), "Table": (0, 0, 255)}

    # Đọc ảnh
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    with st.spinner('🔍 AI is analyzing the drawing structure...'):
        outputs = predictor(image)
        instances = outputs["instances"].to("cpu")
        
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()
        
        final_objects = []
        for i in range(len(instances)):
            if scores[i] < conf_threshold: continue
            
            x1, y1, x2, y2 = map(int, boxes[i])
            label = class_names[classes[i]]
            
            # Crop vùng ảnh để OCR
            crop = image[y1:y2, x1:x2]
            # Tiền xử lý nhẹ cho OCR
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            ocr_res = reader.readtext(gray_crop, detail=0)
            ocr_text = " ".join(ocr_res)
            
            # --- LABEL CORRECTION LOGIC (Phần ăn điểm của bạn) ---
            if label == "Table" and len(ocr_res) < 4 and (x2-x1) < 400:
                label = "Note"
                
            final_objects.append({
                "label": label, 
                "score": float(scores[i]), 
                "bbox": [x1, y1, x2, y2], 
                "ocr": ocr_text
            })

    # --- 5. HIỂN THỊ KẾT QUẢ ---
    col_left, col_right = st.columns([1.5, 1])

    with col_left:
        st.subheader("🖼 Detected Entities")
        display_img = image.copy()
        for obj in final_objects:
            c = colors[obj['label']]
            cv2.rectangle(display_img, (obj['bbox'][0], obj['bbox'][1]), (obj['bbox'][2], obj['bbox'][3]), c, 2)
            # Vẽ label đẹp hơn
            cv2.putText(display_img, f"{obj['label']}", (obj['bbox'][0], obj['bbox'][1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2)
        
        st.image(display_img, channels="BGR", use_container_width=True)

    with col_right:
        st.subheader("📊 Extraction Results")
        
        tab1, tab2 = st.tabs(["📑 OCR Details", "💻 JSON Output"])
        
        with tab1:
            if not final_objects:
                st.warning("No objects detected with current threshold.")
            for idx, obj in enumerate(final_objects):
                with st.expander(f"Object #{idx+1}: {obj['label']} (Conf: {obj['score']:.2f})"):
                    st.write("**Text Content:**")
                    st.info(obj['ocr'] if obj['ocr'].strip() else "Empty text / Handwriting")
        
        with tab2:
            st.json({"file": uploaded_file.name, "count": len(final_objects), "data": final_objects})

else:
    # Màn hình chờ
    st.info("### 👈 Please upload a technical drawing (JPG/PNG) to begin.")
    st.image("https://img.freepik.com/free-vector/blueprint-architecture-concept_23-2147772322.jpg", width=700)
