# 1. Sử dụng Python 3.12.3-slim để tối ưu dung lượng và đồng bộ môi trường
FROM python:3.12.3-slim

# 2. Cài đặt các thư viện hệ thống cần thiết (OpenCV, Git, Build tools)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    git \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. Cài đặt và nâng cấp gdown bản mới nhất
RUN pip install --no-cache-dir --upgrade gdown --break-system-packages

# 4. TẢI MODEL: Sử dụng link trực tiếp để tránh lỗi Exit Code 2
# ID file của bạn: 1voIzijFduwDECvD2OW_OUr4WONvWz9tW
RUN gdown "https://drive.google.com/uc?id=1voIzijFduwDECvD2OW_OUr4WONvWz9tW" -O model_final.pth

# 5. Cài đặt Torch bản CPU (Bắt buộc để không bị tràn RAM trên Render gói Free)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu --break-system-packages

# 6. Cài đặt Detectron2 trực tiếp từ GitHub
RUN pip install --no-cache-dir 'git+https://github.com/facebookresearch/detectron2.git' --break-system-packages

# 7. Cài đặt các thư viện phụ trợ từ requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --break-system-packages

# 8. Copy toàn bộ mã nguồn vào Container
COPY . .

# 9. Mở cổng kết nối cho Streamlit
EXPOSE 8501

# 10. Lệnh khởi chạy ứng dụng
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
