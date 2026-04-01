import os
import cv2
import random
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

# 1. ĐƯỜNG DẪN (Tự động nhận diện thư mục dự án)
base_path = os.path.dirname(os.path.abspath(__file__))
DATASET_NAME = "sotatek_test_preview"
JSON_FILE = os.path.join(base_path, "dataset/train_coco.json")
IMG_DIR = os.path.join(base_path, "dataset")

# Đăng ký metadata để hiện tên lớp (Table, Note, PartDrawing)
if DATASET_NAME not in DatasetCatalog.list():
    register_coco_instances(DATASET_NAME, {}, JSON_FILE, IMG_DIR)
metadata = MetadataCatalog.get(DATASET_NAME)

# 2. CẤU HÌNH (Dùng file đã train xong)
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3 
cfg.MODEL.WEIGHTS = os.path.join(base_path, "output/model_final.pth") 
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
cfg.MODEL.DEVICE = "cpu"

predictor = DefaultPredictor(cfg)

# 3. CHỌN ẢNH NGẪU NHIÊN VÀ LƯU KẾT QUẢ
img_files = [f for f in os.listdir(IMG_DIR) if f.endswith(('.jpg', '.png'))]
test_img_path = os.path.join(IMG_DIR, random.choice(img_files))

im = cv2.imread(test_img_path)
outputs = predictor(im)

# Vẽ kết quả
v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

# LƯU THÀNH FILE ẢNH ĐỂ XEM (Vì máy ảo đôi khi không hiện cửa sổ)
cv2.imwrite("result.jpg", out.get_image()[:, :, ::-1])
print(f"--- ĐÃ XONG! Hãy mở file 'result.jpg' trong thư mục để xem kết quả ---")