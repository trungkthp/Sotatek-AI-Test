import streamlit as st
import cv2
import numpy as np
import os
import time
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import easyocr

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(page_title="Sotatek AI Drawing Dashboard", layout="wide", initial_sidebar_state="expanded")

# CSS tùy chỉnh để giao diện trông hiện đại hơn
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stBrief { font-size: 20px; font-weight: bold; }
    .css-10trblm { color: #1f77b4; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOAD MODELS ---
@st.cache_resource
def load_models():
    base_path = os.path.dirname(os.path.abspath(__file__))
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.WEIGHTS = os.path.join(base_path, "output/model_final.pth")
    cfg.MODEL.DEVICE = "cpu"
    
    predictor = DefaultPredictor(cfg)
    reader = easyocr.Reader(['en'])
    return predictor, reader

# --- 3. SIDEBAR ---
with st.sidebar:
    st.image("https://sotatek.com/wp-content/uploads/2022/04/Sotatek-Logo-Horizontal-White.png", width=200) # Link logo minh họa
    st.title("🛠 Control Panel")
    st.info("Project: Object Detection & OCR for Technical Drawings")
    st.markdown("---")
    uploaded_file = st.file_uploader("📤 Upload Engineering Drawing", type=["jpg", "png", "jpeg"])
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)
    st.markdown("---")
    st.write("Candidate: **[Tên của bạn]**")

# --- 4. XỬ LÝ CHÍNH ---
if uploaded_file is not None:
    predictor, reader = load_models()
    class_names = ["PartDrawing", "Note", "Table"]
    colors = {"PartDrawing": (255, 100, 0), "Note": (0, 200, 100), "Table": (50, 50, 255)}

    # Đọc ảnh
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    with st.spinner('🚀 AI is analyzing the drawing... Please wait.'):
        # Giả lập thời gian chạy một chút cho chuyên nghiệp
        outputs = predictor(image)
        instances = outputs["instances"].to("cpu")
        
        # Lọc logic cao cấp
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()
        
        final_objects = []
        for i in range(len(instances)):
            if scores[i] < conf_threshold: continue
            
            x1, y1, x2, y2 = map(int, boxes[i])
            label = class_names[classes[i]]
            
            # OCR logic
            crop = image[y1:y2, x1:x2]
            ocr_res = reader.readtext(crop, detail=0)
            ocr_text = " ".join(ocr_res)
            
            # Label Correction Logic
            if label == "Table" and len(ocr_res) < 4 and (x2-x1) < 400:
                label = "Note"
                
            final_objects.append({"label": label, "score": scores[i], "bbox": [x1, y1, x2, y2], "ocr": ocr_text})

    # --- 5. HIỂN THỊ KẾT QUẢ ---
    col_left, col_right = st.columns([1.5, 1])

    with col_left:
        st.subheader("🖼 Detected Objects")
        display_img = image.copy()
        for obj in final_objects:
            color = colors[obj['label']]
            cv2.rectangle(display_img, (obj['bbox'][0], obj['bbox'][1]), (obj['bbox'][2], obj['bbox'][3]), color, 3)
            # Vẽ label nền đặc cho dễ đọc
            label_txt = f"{obj['label']} ({obj['score']:.2f})"
            cv2.putText(display_img, label_txt, (obj['bbox'][0], obj['bbox'][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        st.image(display_img, channels="BGR", use_column_width=True)

    with col_right:
        st.subheader("📑 Data Extraction")
        
        # Tab cho gọn
        tab1, tab2 = st.tabs(["OCR Details", "Raw JSON"])
        
        with tab1:
            for idx, obj in enumerate(final_objects):
                with st.expander(f"🔹 {obj['label']} #{idx+1} - Conf: {obj['score']:.2f}"):
                    st.write(f"**Text Content:**")
                    st.code(obj['ocr'] if obj['ocr'] else "No text detected")
        
        with tab2:
            st.json({"image": uploaded_file.name, "results": final_objects})

else:
    # Màn hình chờ khi chưa upload
    st.write("### 👈 Please upload a drawing to start the analysis.")
    st.image("https://img.freepik.com/free-vector/blueprint-architecture-concept_23-2147772322.jpg", width=600)