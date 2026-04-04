import streamlit as st
import cv2
import numpy as np
import os
import subprocess
import sys

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="Sotatek AI Drawing Dashboard", 
    layout="wide"
)

# --- 2. HÀM CÀI ĐẶT & LOAD MODEL ---
@st.cache_resource
def load_models():
    # Kiểm tra và cài đặt Detectron2 nếu chưa có (Lazy Install)
    try:
        from detectron2.engine import DefaultPredictor
        from detectron2.config import get_cfg
        from detectron2 import model_zoo
    except ImportError:
        st.warning("🛠️ Cấu hình môi trường AI lần đầu (khoảng 2-3 phút)...")
        # Cài đặt bản build sẵn để cực nhanh và không tốn RAM
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.1/detectron2-0.6%2Bcpu-cp310-cp310-linux_x86_64.whl"])
        st.success("✅ Đã cài đặt xong! Đang khởi động lại...")
        st.rerun()

    base_path = os.path.dirname(os.path.abspath(__file__))
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    
    model_path = os.path.join(base_path, "output", "model_final.pth")
    if not os.path.exists(model_path):
        st.error(f"❌ Không tìm thấy file model tại: {model_path}")
        st.stop()
        
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.DEVICE = "cpu" 
    
    predictor = DefaultPredictor(cfg)
    # Import EasyOCR ở đây luôn
    import easyocr
    reader = easyocr.Reader(['en'])
    return predictor, reader

# --- 3. TIẾP TỤC LOGIC CỦA TRUNG ---
# ... (Phần Sidebar và xử lý ảnh giữ nguyên)

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
    predictor, reader = load_models()
    class_names = ["PartDrawing", "Note", "Table"]
    colors = {"PartDrawing": (255, 0, 0), "Note": (0, 150, 0), "Table": (0, 0, 255)}

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
            
            crop = image[y1:y2, x1:x2]
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            ocr_res = reader.readtext(gray_crop, detail=0)
            ocr_text = " ".join(ocr_res)
            
            # Logic sửa label dựa trên OCR
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
            for idx, obj in enumerate(final_objects):
                with st.expander(f"Object #{idx+1}: {obj['label']} (Conf: {obj['score']:.2f})"):
                    st.info(obj['ocr'] if obj['ocr'].strip() else "Empty text")
        
        with tab2:
            st.json({"file": uploaded_file.name, "count": len(final_objects), "data": final_objects})
else:
    st.info("### 👈 Please upload a technical drawing to begin.")
