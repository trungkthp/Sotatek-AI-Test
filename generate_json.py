import os
import json
import cv2
import easyocr
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

# 1. KHỞI TẠO
reader = easyocr.Reader(['en'])
base_path = os.path.dirname(os.path.abspath(__file__))

# Cấu hình Detectron2
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
cfg.MODEL.WEIGHTS = os.path.join(base_path, "output/model_final.pth")
cfg.MODEL.DEVICE = "cpu"
predictor = DefaultPredictor(cfg)

class_names = ["PartDrawing", "Note", "Table"]

# Thư mục chứa ảnh và nơi lưu JSON
img_folder = os.path.join(base_path, "dataset")
output_json_folder = os.path.join(base_path, "json_outputs")
os.makedirs(output_json_folder, exist_ok=True)

# 2. QUÉT TOÀN BỘ THƯ MỤC DATASET
all_images = [f for f in os.listdir(img_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

print(f"--- Tìm thấy {len(all_images)} ảnh. Bắt đầu tạo JSON... ---")

for img_name in all_images:
    img_path = os.path.join(img_folder, img_name)
    im = cv2.imread(img_path)
    if im is None:
        print(f"Bỏ qua file lỗi: {img_name}")
        continue

    outputs = predictor(im)
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor.numpy()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    json_data = {
        "image": img_name,
        "objects": []
    }

    for i in range(len(classes)):
        x1, y1, x2, y2 = map(int, boxes[i])
        label = class_names[classes[i]]
        
        ocr_text = ""
        if label in ["Note", "Table"]:
            crop_img = im[y1:y2, x1:x2]
            ocr_result = reader.readtext(crop_img, detail=0)
            ocr_text = " ".join(ocr_result)

        json_data["objects"].append({
            "id": i + 1,
            "class": label,
            "confidence": round(scores[i], 2),
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "ocr_content": ocr_text
        })

    # Lưu file JSON vào thư mục json_outputs
    json_file_name = img_name.split('.')[0] + ".json"
    with open(os.path.join(output_json_folder, json_file_name), "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)
    
    print(f"Đã lưu: {json_file_name}")

print(f"\n--- XONG! Bạn hãy kiểm tra thư mục 'json_outputs' ---")