import os
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

# 1. CẤU HÌNH MODEL
base_path = os.path.dirname(os.path.abspath(__file__))
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3 
cfg.MODEL.WEIGHTS = os.path.join(base_path, "output/model_final.pth") 
cfg.MODEL.DEVICE = "cpu"
predictor = DefaultPredictor(cfg)

class_names = ["PartDrawing", "Note", "Table"]

# 2. ĐƯỜNG DẪN THƯ MỤC
img_folder = os.path.join(base_path, "dataset")
output_crop_dir = os.path.join(base_path, "extracted_all") # Lưu vào thư mục mới cho gọn
os.makedirs(output_crop_dir, exist_ok=True)

# 3. QUÉT TOÀN BỘ FILE TRONG THƯ MỤC DATASET
all_images = [f for f in os.listdir(img_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

print(f"--- Tìm thấy {len(all_images)} ảnh. Bắt đầu quét và cắt... ---")

for img_name in all_images:
    img_path = os.path.join(img_folder, img_name)
    im = cv2.imread(img_path)
    if im is None: continue # Bỏ qua nếu file lỗi

    outputs = predictor(im)
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor.numpy()
    classes = instances.pred_classes.tolist()

    # Tạo thư mục riêng cho từng ảnh gốc để không bị lộn xộn
    image_save_dir = os.path.join(output_crop_dir, img_name.split('.')[0])
    os.makedirs(image_save_dir, exist_ok=True)

    print(f"Đang xử lý: {img_name} -> tìm thấy {len(boxes)} đối tượng.")

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        label = class_names[classes[i]]
        
        # Cắt ảnh
        crop_img = im[y1:y2, x1:x2]
        
        # Lưu mảnh cắt vào thư mục của ảnh đó
        file_name = f"{label}_{i}.jpg"
        cv2.imwrite(os.path.join(image_save_dir, file_name), crop_img)

print(f"\n--- HOÀN THÀNH! Toàn bộ ảnh đã được cắt và lưu vào thư mục 'extracted_all' ---")