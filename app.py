import streamlit as st
import cv2
import numpy as np
import os
import time

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="Sotatek AI Drawing Dashboard", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Giao diện tùy chỉnh cho chuyên nghiệp
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stAlert { border-radius: 10px; }
    .st-expander { border: 1px solid #e0e0e0; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. HÀM LOAD MODELS (Dùng Cache để tránh tràn RAM) ---
@st.cache_resource
def load_models():
    # Import bên trong hàm để tránh lỗi khởi động nếu môi trường chưa sẵn sàng
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    import easyocr

    base_path = os.path.dirname(os.path.abspath(__file__))
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # PartDrawing, Note, Table
    
    # Đảm bảo file model_final.pth nằm trong thư mục output/ trên GitHub của bạn
    model_path = os.path.join(base_path, "output", "model_final.pth")
    
    if not os.path.exists(model_path):
        st.error(f"❌ Không tìm thấy Model tại: {model_path}")
        st.stop()
        
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.DEVICE = "cpu" # Ép chạy CPU cho môi trường Cloud miễn phí
    
    predictor = DefaultPredictor(cfg)
    reader = easyocr.Reader(['en'])
    return predictor, reader

# --- 3. GIAO DIỆN SIDEBAR ---
with st.sidebar:
    st.title("🚀 Sotatek AI Test")
    st.subheader("Technical Drawing Analysis")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("📤 Tải lên bản vẽ (JPG/PNG)", type=["jpg", "png", "jpeg"])
    conf_threshold = st.slider("Ngưỡng tin cậy (Confidence)", 0.0, 1.0, 0.50)
    
    st.markdown("---")
    st.write("📌 **Thông tin dự án:**")
    st.caption("- Framework: Detectron2 & EasyOCR")
    st.write("👤 Ứng viên: **Trung - AI Engineer**")

# --- 4. XỬ LÝ CHÍNH KHI CÓ ẢNH ---
if uploaded_file is not None:
    predictor, reader = load_models()
    class_names = ["PartDrawing", "Note", "Table"]
    colors = {"PartDrawing": (255, 0, 0), "Note": (0, 150, 0), "Table": (0, 0, 255)}

    # Đọc ảnh từ bộ nhớ đệm
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    with st.spinner('🔍 AI đang phân tích...'):
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
            
            # Cắt vùng ảnh để chạy OCR
            crop = image[y1:y2, x1:x2]
            if crop.size == 0: continue
            
            # OCR trích xuất nội dung chữ
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            ocr_res = reader.readtext(gray_crop, detail=0)
            ocr_text = " ".join(ocr_res)
            
            # --- LOGIC SỬA NHÃN (Tăng độ chính xác thực tế) ---
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
        st.subheader("🖼 Bản vẽ đã nhận diện")
        display_img = image.copy()
        for obj in final_objects:
            c = colors[obj['label']]
            cv2.rectangle(display_img, (obj['bbox'][0], obj['bbox'][1]), (obj['bbox'][2], obj['bbox'][3]), c, 2)
            cv2.putText(display_img, f"{obj['label']}", (obj['bbox'][0], obj['bbox'][1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2)
        
        st.image(display_img, channels="BGR", use_container_width=True)

    with col_right:
        st.subheader("📊 Kết quả trích xuất")
        tab1, tab2 = st.tabs(["📑 Chi tiết OCR", "💻 Dữ liệu JSON"])
        
        with tab1:
            if not final_objects:
                st.warning("Không tìm thấy đối tượng nào.")
            for idx, obj in enumerate(final_objects):
                with st.expander(f"Đối tượng #{idx+1}: {obj['label']} ({obj['score']:.2f})"):
                    st.info(obj['ocr'] if obj['ocr'].strip() else "Trống")
        
        with tab2:
            st.json({"file": uploaded_file.name, "count": len(final_objects), "data": final_objects})
else:
    st.info("### 👈 Vui lòng tải lên bản vẽ để bắt đầu phân tích.")
