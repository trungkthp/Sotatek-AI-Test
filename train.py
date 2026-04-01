import os
import torch
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog

# --- 1. ĐĂNG KÝ DỮ LIỆU ---
# Tự động lấy đường dẫn thư mục hiện tại (Sotatek_Project)
base_path = os.path.dirname(os.path.abspath(__file__))

DATASET_NAME = "sotatek_train"
# Đường dẫn đã được sửa để bao gồm "Bản tải về" một cách linh hoạt
JSON_FILE = os.path.join(base_path, "dataset/train_coco.json")
IMG_DIR = os.path.join(base_path, "dataset")

# Xóa đăng ký cũ nếu có để tránh lỗi khi chạy lại nhiều lần
if DATASET_NAME in DatasetCatalog.list():
    DatasetCatalog.remove(DATASET_NAME)

register_coco_instances(DATASET_NAME, {}, JSON_FILE, IMG_DIR)

# --- 2. CẤU HÌNH MODEL ---
cfg = get_cfg()
# Sử dụng Faster R-CNN R50 FPN
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = (DATASET_NAME,)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2

# Tải weights (bộ não) có sẵn của Facebook để học nhanh hơn
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 500  # Chạy 500 vòng để kịp thời gian
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # Table, Note, PartDrawing

# QUAN TRỌNG: Ép chạy bằng CPU vì máy ảo thường không nhận GPU
cfg.MODEL.DEVICE = "cpu"

# --- 3. BẮT ĐẦU TRAIN ---
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
print("--- ĐANG KHỞI TẠO VÀ BẮT ĐẦU TRAIN, VUI LÒNG ĐỢI ---")
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()