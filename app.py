import streamlit as st
import cv2
import numpy as np
import os
import gc
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import easyocr

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="Sotatek AI Test", layout="wide")
st.title("🚀 Technical Drawing Analysis")

# --- HÀM LOAD MODEL DETECTRON2 (CHỈ LOAD 1 LẦN) ---
@st.cache_resource
def get_predictor():
    base_path = os.path.dirname(os.path.abspath(__file__))
    # Tự động tìm model dù ở gốc hay trong output/
    p1 = os.path.join(base_path, "model_final.pth")
    p2 = os.path.join(base_path, "output", "model_final.pth")
    model_path = p1 if os.path.exists(p1) else p2
    
    if not os.path.exists(model_path):
        st.error(f"❌ Không tìm thấy model tại {p1} hoặc {p2}")
        st.stop()

    cfg = get_cfg()
    # Dùng ResNet-50 để nhẹ RAM hơn ResNet-101
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.DEVICE = "cpu"
    # Giới hạn số lượng proposal để giảm tải tính toán
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300 
    
    return DefaultPredictor(cfg)

# --- GIAO DIỆN UPLOAD ---
uploaded_file = st.file_uploader("Upload bản vẽ kỹ thuật", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # 1. Khởi tạo Predictor (Lần đầu sẽ hơi lâu)
    predictor = get_predictor()
    
    # 2. Đọc ảnh từ file upload
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    with st.spinner('AI đang phân tích đối tượng...'):
        # 3. Chạy Detection
        outputs = predictor(image)
        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()
        classes = instances.pred_classes.numpy()
        scores = instances.scores.numpy()
        
        # 4. Chỉ load EasyOCR SAU KHI detect xong để tiết kiệm RAM
        # Điều này giúp tránh việc 2 model nặng cùng nằm trên RAM một lúc
        with st.spinner('Đang nhận diện chữ viết (OCR)...'):
            reader = easyocr.Reader(['en'], gpu=False)
            
            # --- LOGIC XỬ LÝ KẾT QUẢ (Vẽ và OCR) ---
            for i, box in enumerate(boxes):
                if scores[i] > 0.5: # Confidence threshold
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Cắt vùng ảnh để OCR
                    roi = image[y1:y2, x1:x2]
                    if roi.size > 0:
                        results = reader.readtext(roi)
                        text = results[0][1] if results else ""
                        
                        # Vẽ lên ảnh chính
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(image, text, (x1, y1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # 5. Hiển thị kết quả
        st.image(image, channels="BGR", use_container_width=True)
        st.success(f"Phân tích thành công {len(boxes)} đối tượng!")
        
        # Dọn dẹp bộ nhớ tạm
        del outputs
        gc.collect()
