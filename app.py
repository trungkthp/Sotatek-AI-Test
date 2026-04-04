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
st.set_page_config(
    page_title="Sotatek AI Drawing Dashboard", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Tùy chỉnh giao diện bằng CSS
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stAlert { border-radius: 10px; }
    .st-expander { border: 1px solid #e0e0e0; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOAD MODELS ---
@st.cache_resource
def load_models():
    # Lấy đường dẫn thư mục gốc
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    cfg = get_cfg()
    # Sử dụng cấu hình Faster R-CNN tương ứng với model đã train
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # PartDrawing, Note, Table
    
    # Đường dẫn trỏ vào file weight trong thư mục output/
    model_path = os.path.join(base_path, "output", "model_final.pth")
    
    if not os.path.exists(model_path):
        st.error(f"❌ Không tìm thấy file Model tại: {model_path}")
        st.stop()
        
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.DEVICE = "cpu" # Ép chạy CPU trên Streamlit Cloud
    
    predictor = DefaultPredictor(cfg)
    # Khởi tạo EasyOCR cho tiếng Anh
    reader = easyocr.Reader(['en'])
    return predictor, reader

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("🚀 Sotatek AI Test")
    st.subheader("Technical Drawing Analysis")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("📤 Upload Engineering Drawing", type=["jpg", "png", "jpeg"])
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.50)
    
    st.markdown("---")
    st.write("📌 **Project Information:**")
    st.caption("- Framework: Detectron2 & EasyOCR")
    st.write("👤 Candidate: **Trung - AI Engineer**")

# --- 4. XỬ LÝ CHÍNH ---
if uploaded_file is not None:
    # Gọi hàm load model (sẽ dùng cache sau lần đầu)
    predictor, reader = load_models()
    
    class_names = ["PartDrawing", "Note", "Table"]
    colors = {"PartDrawing": (255, 0, 0), "Note": (0, 150, 0), "Table": (0, 0, 255)}

    # Đọc ảnh từ file upload
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    with st.spinner('🔍 AI đang phân tích cấu trúc bản vẽ...'):
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
            
            # Crop vùng đối tượng để OCR
            crop = image[y1:y2, x1:x2]
            if crop.size == 0: continue
            
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            ocr_res = reader.readtext(gray_crop, detail=0)
            ocr_text = " ".join(ocr_res)
            
            # --- LOGIC SỬA LABEL (Dựa trên kết quả OCR) ---
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
            cv2.putText(display_img, f"{obj['label']}", (obj['bbox'][0], obj['bbox'][1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2)
        
        st.image(display_img, channels="BGR", use_container_width=True)

    with col_right:
        st.subheader("📊 Extraction Results")
        tab1, tab2 = st.tabs(["📑 OCR Details", "💻 JSON Output"])
        
        with tab1:
            if not final_objects:
                st.warning("Không tìm thấy đối tượng nào với ngưỡng Confidence hiện tại.")
            for idx, obj in enumerate(final_objects):
                with st.expander(f"Object #{idx+1}: {obj['label']} ({obj['score']:.2f})"):
                    st.write("**Nội dung OCR:**")
                    st.info(obj['ocr'] if obj['ocr'].strip() else "Không nhận diện được chữ")
        
        with tab2:
            st.json({"file": uploaded_file.name, "count": len(final_objects), "data": final_objects})

else:
    st.info("### 👈 Vui lòng tải lên bản vẽ kỹ thuật (JPG/PNG) để bắt đầu.")
